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
  MemorySaver,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";


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

// const modelWithTools = new ChatGoogleGenerativeAI({
//     model: process.env.MODEL_NAME!,
//     apiKey: process.env.GEMINI_API_KEY!,
//     temperature: 0,
// }).bindTools(tools);

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
  const lastMessage = state.messages.at(-1);

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
  const lastMessage = state.messages.at(-1);

  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  return END;
};

const checkpointer = new MemorySaver();
const config = { configurable: { thread_id: '1' }} as LangGraphRunnableConfig;

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile({ checkpointer });


console.log("Use '.exit' to quit!");
let lineNo = 0;

const reader = console[Symbol.asyncIterator]();
while (true) {
  process.stdout.write("[HUMAN]: ");
  const line = ((await reader.next()).value as string).trim();
  if (line === '.exit') break;

  const messages = [new HumanMessage(line)];
  const result = await agent.invoke({ messages }, config);

  let lines = 0;
  for (const msg of result.messages.slice(lineNo)) {
    if (msg.type === "ai" && msg.text.trim().length > 0) {
      console.log(`[${msg.type.toUpperCase()}]: ${msg.text}`);
    }
    lines += 1;
  }
  lineNo += lines;
}

await reader.return?.();

// const state = await agent.getState(config);
// console.log("State:", state);
