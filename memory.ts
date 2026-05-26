import {
  InMemoryStore,
  MemorySaver,
  StateGraph,
  START,
  END,
  type LangGraphRunnableConfig,
  Annotation,
  MessagesAnnotation
} from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

/**
 * memory.ts
 *
 * SIMPLE EXAMPLE OF MEMORY STORE (BaseStore) IN LANGGRAPH
 *
 * Documentation: https://docs.langchain.com/oss/javascript/langgraph/persistence#memory-store
 *
 * InMemoryStore allows for long-term, cross-thread memory (e.g., user profiles, preferences).
 * This is different from Checkpointers (MemorySaver) which only persist a single thread.
 */

// 1. Initialize the model (using Ollama as requested)
const model = new ChatOllama({
  model: "gemma4:e2b",
  temperature: 0.7,
  baseUrl: "http://127.0.0.1:11434",
});

// 2. Initialize the InMemoryStore for cross-thread persistence
const store = new InMemoryStore();

// 3. Initialize the MemorySaver for single-thread persistence (checkpointer)
const checkpointer = new MemorySaver();

// 4. Define the Graph State
const GraphState = Annotation.Root({
  ...MessagesAnnotation.spec,
});

// 5. Define the Agent Node (Generates response)
const agentNode = async (state: typeof GraphState.State, config: LangGraphRunnableConfig) => {
  const userId = config.configurable?.user_id || "default_user";
  const namespace = ["memories", userId];

  // Retrieve memories from the store
  const memories = await config.store?.search(namespace);
  const info = memories && memories.length > 0 
    ? memories.map((m) => `- ${m.value.content}`).join("\n") 
    : "No previous information known about this user.";

  console.log(`\n[System] Context retrieved from Store for ${userId}:\n${info}`);

  const systemMessage = new SystemMessage(
    `You are a helpful assistant. Here is context from PREVIOUS conversations:\n${info}\n\nRespond to the user briefly.`
  );

  const response = await model.invoke([systemMessage, ...state.messages]);
  return { messages: [response] };
};

// 6. Define the Memory Management Node (Extracts and saves memories)
const manageMemory = async (state: typeof GraphState.State, config: LangGraphRunnableConfig) => {
  const userId = config.configurable?.user_id || "default_user";
  const namespace = ["memories", userId];

  // We look at the last exchange to see if there's anything worth remembering
  const lastMessages = state.messages.slice(-2); 
  
  const extractionPrompt = new SystemMessage(
    `Analyze the conversation and extract any NEW permanent user preferences or facts.
    If there is something new to remember, respond ONLY with the fact in a single short sentence.
    Example: "The user likes espresso."
    If there is nothing new to remember, respond with "NONE".`
  );

  const extractionResponse = await model.invoke([extractionPrompt, ...lastMessages]);
  const content = extractionResponse.content as string;

  if (content.toUpperCase().trim() !== "NONE" && content.trim().length > 0) {
    console.log(`[System] LLM identified new memory to save: "${content}"`);
    const memoryId = crypto.randomUUID();
    await config.store?.put(namespace, memoryId, { content: content });
  }

  return {}; 
};

// 7. Build the Graph
const workflow = new StateGraph(GraphState)
  .addNode("agent", agentNode)
  .addNode("manageMemory", manageMemory)
  .addEdge(START, "agent")
  .addEdge("agent", "manageMemory") // Run memory management after responding
  .addEdge("manageMemory", END);

// 8. Compile the graph with BOTH checkpointer and store
const app = workflow.compile({ checkpointer, store });

// 8. Run the example to demonstrate cross-thread memory
async function run() {
  const userId = "user_abc_123";

  console.log("=== SESSION 1 (Thread: thread_alpha) ===");
  const config1 = {
    configurable: {
      thread_id: "thread_alpha",
      user_id: userId
    }
  };

  console.log("Human: Hi! I like cold brew coffee.");
  const output1 = await app.invoke({
    messages: [new HumanMessage("Hi! I like cold brew coffee.")]
  }, config1);

  const lastMsg1 = output1.messages[output1.messages.length - 1] as AIMessage;
  console.log("Assistant:", lastMsg1.content);

  console.log("\n-------------------------------------------");

  console.log("=== SESSION 2 (Thread: thread_beta - NEW THREAD) ===");
  // We use a DIFFERENT thread_id, so the checkpointer won't have the previous messages.
  // However, because it's the SAME user_id, the MemoryStore will provide the context.
  const config2 = {
    configurable: {
      thread_id: "thread_beta",
      user_id: userId
    }
  };

  console.log("Human: What drink do I like?");
  const output2 = await app.invoke({ 
    messages: [new HumanMessage("What drink do I like?")] 
  }, config2);

  const lastMsg2 = output2.messages[output2.messages.length - 1] as AIMessage;
  console.log("Assistant:", lastMsg2.content);

  console.log("\n=== STORE INSPECTION ===");
  // We can "look into" the store by searching a namespace.
  // To see everything in the "memories" root namespace:
  const allMemories = await store.search(["memories"]);
  console.log(`Found ${allMemories.length} items in the store:`);

  allMemories.forEach((item, index) => {
    console.log(`\n[Item ${index + 1}]`);
    console.log(`Namespace: ${JSON.stringify(item.namespace)}`);
    console.log(`Key:       ${item.key}`);
    console.log(`Value:     ${JSON.stringify(item.value)}`);
  });
  }

  run().catch((err) => {  console.error("Error running memory example:", err);
  process.exit(1);
});
