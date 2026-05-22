import { ChatOllama } from "@langchain/ollama";

const model = new ChatOllama({
  model: "gemma4:e2b",
  temperature: 0.7,
  baseUrl: "http://127.0.0.1:11434",
});

console.log("Asking local Gemma model...");

const response = await model.invoke([
  { role: "user", content: "What are the advantages of testing an agent loop locally?" }
]);

console.log("\nResponse:\n", response.content);
