import { tool } from "@langchain/core/tools";
import { SystemMessage, AIMessage, ToolMessage, HumanMessage } from "@langchain/core/messages";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  ReducedValue,
  type GraphNode,
  type ConditionalEdgeRouter,
  START,
  END,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { BunSqliteSaver } from "langgraph-checkpoint-bunsqlite";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";

/*
// Previous manual file-based persistence (Commented out)
const HISTORY_FILE = "./chat_history.json";

async function saveHistory(messages: any[]) {
  const data = JSON.stringify(messages, null, 2);
  await Bun.write(HISTORY_FILE, data);
}

async function loadHistory() {
  const file = Bun.file(HISTORY_FILE);
  if (await file.exists()) {
    const raw = await file.json();
    return raw.map((m: any) => {
      if (m.type === "human") return new HumanMessage(m.content);
      if (m.type === "ai") return new AIMessage({ content: m.content, tool_calls: m.tool_calls });
      if (m.type === "system") return new SystemMessage(m.content);
      if (m.type === "tool") return new ToolMessage({ content: m.content, tool_call_id: m.tool_call_id });
      return m;
    });
  }
  return [];
}
*/

const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const toolsByName = {
    [add.name]: add,
    [multiply.name]: multiply,
    [divide.name]: divide,
};
type ToolName = keyof typeof toolsByName;
const tools = Object.values(toolsByName);

const modelWithTools = new ChatOllama({
  model: "gemma4:e2b",
  baseUrl: "http://127.0.0.1:11434",
  temperature: 0.7,
}).bindTools(tools);

const MessagesState = new StateSchema({
  messages: MessagesValue,
  llmCalls: new ReducedValue(
    z.number().default(0),
    { reducer: (x, y) => x + y }
  ),
});

const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    ),
    ...state.messages,
  ]);
  return {
    messages: [response],
    llmCalls: 1,
  };
};

const toolNode: GraphNode<typeof MessagesState> = async (state) => {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
    return { messages: [] };
  }

  const result: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name as ToolName];
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return { messages: result };
};

const shouldContinue: ConditionalEdgeRouter<typeof MessagesState, Record<string, any>, "toolNode"> = (state) => {
  const lastMessage = state.messages[state.messages.length - 1];

  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  return END;
};

// --- BUN SQLITE PERSISTENCE ---
const checkpointer = new BunSqliteSaver({ dbPath: "./checkpoints.db" });
const config = { configurable: { thread_id: '1' }} as LangGraphRunnableConfig;

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile({ checkpointer });

// --- RESUME LOGIC (Bun SQLite) ---
const existingState = await agent.getState(config);

if (existingState.values.messages && (existingState.values.messages as any[]).length > 0) {
  console.log(`\n[System]: Resuming session '${config.configurable?.thread_id}' from Bun SQLite DB...`);
  const history = existingState.values.messages as any[];

  for (const msg of history) {
    if (msg.content && msg.content.trim().length > 0) {
      console.log(`[${msg.type.toUpperCase()}]: ${msg.content}`);
    }
  }
  console.log("[System]: History loaded. You can continue the chat.\n");
} else {
  console.log("\n[System]: Started a new session.");
  console.log("Use '.exit' to quit!\n");
}

const reader = console[Symbol.asyncIterator]();
while (true) {
  process.stdout.write("[HUMAN]: ");
  const line = ((await reader.next()).value as string).trim();
  if (line === '.exit') break;

  const messages = [new HumanMessage(line)];
  const stream = await agent.stream(
    { messages },
    { ...config, streamMode: "messages" }
  );

  let aiHeaderPrinted = false;
  for await (const [msg, metadata] of stream) {
    if (msg.type === "ai" && metadata.langgraph_node === "llmCall") {
      if (!aiHeaderPrinted) {
        process.stdout.write("[AI]: ");
        aiHeaderPrinted = true;
      }
      if (msg.content) {
        process.stdout.write(msg.content as string);
      }
    }
  }

  if (aiHeaderPrinted) {
    process.stdout.write("\n");
  }

  // Automatic checkpointing happens here via the checkpointer
}

await reader.return?.();
