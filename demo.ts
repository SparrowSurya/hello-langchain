/**
 * demo.ts
 * 
 * PRODUCTION-GRADE LANGGRAPH ARCHITECTURE
 * 
 * This file demonstrates the architectural "missing pieces" required to take 
 * a prototype (like example.ts) into a real-world production environment.
 */

import { z } from "zod";
import { StateGraph, START, END, Annotation, messagesStateReducer, MemorySaver } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

/**
 * 1. PERSISTENCE & THREAD MANAGEMENT
 * In production, you don't just run a script. You have users with "threads".
 * We use Checkpointers (like PostgresSaver or MemorySaver) to save state 
 * after every node. This allows:
 * - Resuming a conversation after hours/days.
 * - Multi-turn interaction (e.g., user says "now scale it for 10").
 * - Error recovery (resume from the last successful node).
 */
const checkpointer = new MemorySaver();

/**
 * 2. REFINED STATE & VERSIONING
 * Production state needs to track metadata, user IDs, and "snapshots".
 */
const ProState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  // Metadata for tracing and analytics
  metadata: Annotation<{
    userId: string;
    threadId: string;
    startTime: number;
  }>({
    reducer: (x, y) => ({ ...x, ...y }),
  }),
  // Structured data with internal versioning or "last validated" flags
  data: Annotation<{
    recipe?: any;
    isValidated: boolean;
    retryCount: number;
  }>({
    reducer: (x, y) => ({ ...x, ...y }),
    default: () => ({ isValidated: false, retryCount: 0 }),
  }),
});

/**
 * 3. EXTERNAL API INTEGRATION (SIDE EFFECTS)
 * Production nodes rarely just call LLMs. They call:
 * - Vector DBs (Retrieval)
 * - SQL Databases (User history/Inventory)
 * - External APIs (Grocery pricing/availability)
 */
async function fetchInventoryNode(state: typeof ProState.State) {
  // Logic to check real-time stock from a DB or API
  console.log("Checking real-world inventory...");
  return { data: { ...state.data, inventoryChecked: true } };
}

/**
 * 4. HUMAN-IN-THE-LOOP (HITL)
 * Crucial for sensitive tasks (like spending money or large quantities).
 * We can "interrupt" the graph and wait for a human to approve.
 */
async function approvalNode(state: typeof ProState.State) {
  // This node marks that we need human approval
  // The graph will STOP here if we configure an interrupt
  return { data: { ...state.data, isValidated: true } };
}

/**
 * 5. ERROR HANDLING & AUTO-RETRIES
 * Production systems use "fallback" patterns. If an LLM fails 
 * or returns invalid JSON, we route to a "Fixer" node or retry with a different model.
 */
async function extractionNode(state: typeof ProState.State) {
  try {
    // LLM call here...
    return { data: { ...state.data, retryCount: 0 } };
  } catch (e) {
    if (state.data.retryCount < 3) {
      console.warn("Retrying extraction...");
      return { data: { ...state.data, retryCount: state.data.retryCount + 1 } };
    }
    throw e; // Escalation
  }
}

/**
 * 6. OBSERVABILITY (TRACING)
 * Not code-based, but crucial. You connect this to LangSmith.
 * It tracks:
 * - Latency per node.
 * - Token cost per user.
 * - Quality scores via human feedback.
 */

// ════════════════════════════════════════════════════════════
//  THE PRODUCTION GRAPH
// ════════════════════════════════════════════════════════════

const workflow = new StateGraph(ProState)
  .addNode("extract", extractionNode)
  .addNode("inventory", fetchInventoryNode)
  .addNode("approve", approvalNode)
  
  .addEdge(START, "extract")
  .addEdge("extract", "inventory")
  .addEdge("inventory", "approve")
  .addEdge("approve", END);

/**
 * 7. DEPLOYMENT & STREAMING
 * In production, you don't wait for the final result. You STREAM:
 * - token-by-token (for UI responsiveness)
 * - node-by-node (to show progress bars)
 */

export const app = workflow.compile({
  checkpointer,
  // 8. INTERRUPTS
  // The graph will pause BEFORE the 'approve' node and wait for external input
  interruptBefore: ["approve"], 
});

/**
 * SUMMARY OF MISSING PARTS IN example.ts:
 * 
 * 1. PERSISTENCE: example.ts is stateless. Production needs DB-backed state.
 * 2. AUTH/IDENTITY: No concept of which user is asking.
 * 3. COST MANAGEMENT: No tracking of how many tokens are used.
 * 4. RATE LIMITING: No handling of LLM provider limits (429 errors).
 * 5. PROMPT VERSIONING: Prompts are hardcoded strings. Production uses 
 *    a Prompt Registry (like LangSmith) to update prompts without redeploying code.
 * 6. TESTING (EVALS): example.ts is tested manually. Production uses 
 *    "Evaluation Sets" to run 100s of inputs and check for regressions automatically.
 * 7. STREAMING: example.ts uses 'await pipeline.invoke', which is "blocking".
 *    Production uses '.stream()' to provide a better UX.
 */

// Example of how to trigger a production run:
/*
const config = { configurable: { thread_id: "user-123" } };
await app.invoke({ metadata: { userId: "user-123", threadId: "abc" } }, config);

// To resume after human approval:
// await app.invoke(null, config); 
*/
