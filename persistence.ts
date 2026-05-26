/**
 * memory.ts
 * 
 * ADVANCED PERSISTENCE & TIME TRAVEL IN LANGGRAPH
 * 
 * This file demonstrates the core persistence features based on the official 
 * LangChain documentation: https://docs.langchain.com/oss/javascript/langgraph/persistence
 */

import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";

// 1. STATE DEFINITION
const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  count: Annotation<number>({
    reducer: (x, y) => x + y, // Incremental reducer
    default: () => 0,
  }),
});

// 2. NODES
const callModel = async (state: typeof State.State) => {
  console.log(`--- Executing Node: count is ${state.count} ---`);
  return { 
    messages: [new AIMessage(`Step ${state.count + 1}`)],
    count: 1 // This will be ADDED to state.count because of the reducer
  };
};

// 3. GRAPH COMPILATION WITH PERSISTENCE
const checkpointer = new MemorySaver();
const workflow = new StateGraph(State)
  .addNode("agent", callModel)
  .addEdge(START, "agent")
  .addEdge("agent", END);

const app = workflow.compile({ checkpointer });

async function runPersistenceDemo() {
  const config = { configurable: { thread_id: "demo-thread-001" } };

  console.log("\n=== 1. BASIC THREAD PERSISTENCE ===");
  await app.invoke({ messages: [new HumanMessage("Hello")], count: 0 }, config);
  await app.invoke({ messages: [new HumanMessage("Second msg")], count: 0 }, config);

  // 4. INSPECTING STATE (getState)
  console.log("\n=== 2. INSPECTING CURRENT STATE ===");
  const currentState = await app.getState(config);
  console.log("Current Count:", currentState.values.count);
  console.log("Next node to run:", currentState.next); // Should be empty if graph ended

  // 5. MANUAL STATE UPDATE (updateState)
  console.log("\n=== 3. MANUAL STATE UPDATE ===");
  // updateState creates a NEW checkpoint. 
  // Because 'count' has a reducer (x + y), this will ADD 10 to the state.
  await app.updateState(config, { count: 10 });
  const updatedState = await app.getState(config);
  console.log("Updated Count (2 + 10):", updatedState.values.count);

  // 6. CHECKPOINT HISTORY
  console.log("\n=== 4. VIEWING HISTORY (SNAPSHOTS) ===");
  const history = [];
  for await (const state of app.getStateHistory(config)) {
    console.log(`ID: ${state.config.configurable.checkpoint_id.slice(0, 8)} | Count: ${state.values.count}`);
    history.push(state);
  }

  // 7. TIME TRAVEL (REPLAYING)
  console.log("\n=== 5. TIME TRAVEL (REPLAY FROM PAST) ===");
  if (history.length > 2) {
    // Let's pick an older checkpoint (from before we added 10)
    const pastCheckpoint = history[history.length - 1]; 
    const replayConfig = {
      configurable: {
        thread_id: "demo-thread-001",
        checkpoint_id: pastCheckpoint.config.configurable.checkpoint_id
      }
    };
    
    console.log(`Replaying from checkpoint with count: ${pastCheckpoint.values.count}`);
    // This will resume execution from that specific state
    const result = await app.invoke(null, replayConfig);
    console.log("Replay Result count:", result.count);
  }
}

runPersistenceDemo().catch(console.error);

/**
 * PRODUCTION TAKEAWAYS:
 * 
 * 1. THREADS: Isolation between different users/conversations.
 * 2. CHECKPOINTS: Snapshots at every 'super-step' boundary.
 * 3. IMMUTABILITY: updateState doesn't delete history; it branches it.
 * 4. TIME TRAVEL: You can "Rewind" an agent by passing a checkpoint_id 
 *    in the configuration. This is vital for "Undo" features in UIs.
 * 5. BRANCHING: You can modify a past state and resume, creating a 
 *    different future for the agent (A/B testing for trajectories).
 */
