/**
 * snapcart-pipeline.ts
 *
 * Recipe text → structured ingredient list using LangGraph + Gemini / Ollama.
 *
 * Run:
 *   npx tsx snapcart-pipeline.ts
 *   npx tsx snapcart-pipeline.ts "Paneer Butter Masala for 4"
 *   MODEL_PROVIDER=ollama npx tsx snapcart-pipeline.ts "Dal Makhani for 2"
 *
 * Install dependencies first:
 *   npm install @langchain/core @langchain/langgraph @langchain/google-genai @langchain/ollama zod dotenv
 */

import { z } from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOllama } from "@langchain/ollama";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, START, END, messagesStateReducer } from "@langchain/langgraph";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";

// ════════════════════════════════════════════════════════════
//  CONFIG — edit these or use environment variables
// ════════════════════════════════════════════════════════════

const CONFIG = {
  // "gemini" or "ollama"
  provider: (process.env.MODEL_PROVIDER ?? "ollama") as "gemini" | "ollama",

  gemini: {
    apiKey: process.env.GEMINI_API_KEY ?? "",
    model:  process.env.GEMINI_MODEL  ?? "gemini-1.5-flash",
  },

  ollama: {
    baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
    model:   process.env.OLLAMA_MODEL   ?? "gemma4:e2b",
  },
};

// ════════════════════════════════════════════════════════════
//  MODEL FACTORY
// ════════════════════════════════════════════════════════════

function createModel(): BaseChatModel {
  if (CONFIG.provider === "ollama") {
    return new ChatOllama({
      model:   CONFIG.ollama.model,
      baseUrl: CONFIG.ollama.baseUrl,
      temperature: 0,
      format: "json", // required for reliable structured output with Ollama
    });
  }

  if (!CONFIG.gemini.apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not set.\n" +
      "Either set it in a .env file or run: GEMINI_API_KEY=your_key npx tsx snapcart-pipeline.ts"
    );
  }

  return new ChatGoogleGenerativeAI({
    model:      CONFIG.gemini.model,
    apiKey:     CONFIG.gemini.apiKey,
    temperature: 0,
  });
}

// ════════════════════════════════════════════════════════════
//  ZOD SCHEMAS  (structured output contracts for each node)
// ════════════════════════════════════════════════════════════

// ── Step 0: Intent Detection ─────────────────────────────────────────
const IntentSchema = z.object({
  isRecipeIntent: z.boolean()
    .describe("True only if the user is asking for a recipe, dish ingredients, or meal planning."),
  reason: z.string()
    .describe("Brief explanation of why this is or isn't a recipe-related request."),
});

// ── Step 1: Recipe Identification ────────────────────────────────────
const ParsedRecipeSchema = z.object({
  recipeName: z.string()
    .describe("Canonical recipe name, properly capitalized. e.g. 'Butter Chicken'"),
  servings: z.number().int().min(1)
    .describe("Number of servings. Default to 2 if not mentioned."),
  cuisine: z.string()
    .describe("Cuisine type. e.g. 'Indian', 'Italian', 'Mexican'"),
  confidence: z.number().min(0).max(1)
    .describe("Confidence in the identification from 0 to 1."),
});

// ── Step 2: Ingredient Finding ───────────────────────────────────────
const RawIngredientSchema = z.object({
  name:     z.string().describe("Ingredient name in plain English"),
  quantity: z.number().describe("Base numeric quantity (unscaled or for 1 serving)"),
  unit:     z.string().describe("Unit: grams, ml, pieces, tbsp, tsp, cups, etc."),
  category: z.enum([
    "meat_seafood",
    "vegetable_fruit",
    "dairy",
    "spice_herb",
    "grain_flour",
    "oil_fat",
    "sauce_condiment",
    "other",
  ]),
  optional: z.boolean().describe("True if optional/garnish"),
});

const IngredientsListSchema = z.object({
  ingredients: z.array(RawIngredientSchema),
});

// ── Step 3: Validation ───────────────────────────────────────────────
const ValidationSchema = z.object({
  isProper: z.boolean()
    .describe("True if these ingredients accurately represent the canonical recipe."),
  matchesTasteProfile: z.boolean()
    .describe("True if the ingredients align with user preferences (e.g. extra spicy, low salt, etc.)."),
  feedback: z.string()
    .describe("A clear explanation of why the ingredients are proper or if something is missing/mismatched."),
});

// ── Step 4: Final Normalized Ingredient ──────────────────────────────
const NormalizedIngredientSchema = z.object({
  name:        z.string(),
  quantity:    z.number(),
  unit:        z.string(),
  category:    z.string(),
  optional:    z.boolean(),
  purchasable: z.boolean()
    .describe("False only for common household staples like water, salt, or plain oil."),
  searchQuery: z.string()
    .describe("Best search term for an Indian grocery app."),
});

const NormalizedListSchema = z.object({
  ingredients: z.array(NormalizedIngredientSchema),
});

// TypeScript types
type RawIngredient        = z.infer<typeof RawIngredientSchema>;
type NormalizedIngredient = z.infer<typeof NormalizedIngredientSchema>;

// ════════════════════════════════════════════════════════════
//  LANGGRAPH STATE
// ════════════════════════════════════════════════════════════

const RecipeState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),

  rawInput: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),

  // Intent
  isRecipeIntent: Annotation<boolean>({
    reducer: (_, y) => y,
    default: () => false,
  }),
  intentReason: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),

  // Recipe Identification
  recipeName: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),
  servings: Annotation<number>({
    reducer: (_, y) => y,
    default: () => 2,
  }),
  cuisine: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),
  confidence: Annotation<number>({
    reducer: (_, y) => y,
    default: () => 0,
  }),

  // Ingredients & Validation
  rawIngredients: Annotation<RawIngredient[]>({
    reducer: (_, y) => y,
    default: () => [],
  }),
  isProper: Annotation<boolean>({
    reducer: (_, y) => y,
    default: () => false,
  }),
  tasteMatch: Annotation<boolean>({
    reducer: (_, y) => y,
    default: () => false,
  }),
  validationFeedback: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),

  // Final Output
  finalIngredients: Annotation<NormalizedIngredient[]>({
    reducer: (_, y) => y,
    default: () => [],
  }),

  exitMessage: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "",
  }),

  currentStep: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "idle",
  }),

  errors: Annotation<string[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
});

// ════════════════════════════════════════════════════════════
//  NODES
// ════════════════════════════════════════════════════════════

const model = createModel();

// ── Node 0: Detect Intent ──────────────────────────────────────────
async function detectIntentNode(
  state: typeof RecipeState.State
): Promise<typeof RecipeState.Update> {
  console.log("\n┌─ Step 0  Detecting user intent");
  try {
    const structured = model.withStructuredOutput(IntentSchema, { name: "detect_intent" });
    const result = await structured.invoke([
      new SystemMessage(
        `You are a strict intent classifier for a recipe assistant.
Determine if the user's input is a request for a recipe, dish ingredients, meal planning, or a request to prepare a specific dish (even if the word 'recipe' isn't used).

Rules:
- VALID Intent: "Paneer Butter Masala for 4", "Tea for 50", "How to make a cake", "What do I need for burgers?".
- INVALID Intent: "How are you?", "Tell me a joke", "Write code", "Weather in NYC", "Who is the president?".
- Be precise: If a dish name and a quantity are mentioned, it is ALMOST ALWAYS a request for ingredients/quantities for that dish.`
      ),
      new HumanMessage(state.rawInput),
    ]);

    console.log(`│  Is Recipe Intent : ${result.isRecipeIntent}`);
    console.log(`└  Reason           : ${result.reason}`);

    if (!result.isRecipeIntent) {
      return {
        isRecipeIntent: false,
        intentReason: result.reason,
        currentStep: "intent_rejected",
        exitMessage: `User intention does not match: ${result.reason}`,
      };
    }

    return {
      isRecipeIntent: true,
      intentReason: result.reason,
      currentStep: "intent_detected",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { errors: [`detectIntentNode: ${msg}`], currentStep: "intent_failed" };
  }
}

// ── Node 1: Identify Recipe ─────────────────────────────────────────
async function identifyRecipeNode(
  state: typeof RecipeState.State
): Promise<typeof RecipeState.Update> {
  console.log("\n┌─ Step 1  Identifying recipe");
  try {
    const structured = model.withStructuredOutput(ParsedRecipeSchema, { name: "identify_recipe" });
    const result = await structured.invoke([
      new SystemMessage(
        `Extract recipe details from the input.
Rules:
- Return canonical recipe name and cuisine.
- Identify serving count (default to 2).
- Set confidence (0-1). Use < 0.7 if the dish is unknown or input is vague.`
      ),
      new HumanMessage(state.rawInput),
    ]);

    console.log(`│  Recipe     : ${result.recipeName}`);
    console.log(`│  Cuisine    : ${result.cuisine}`);
    console.log(`└  Confidence : ${(result.confidence * 100).toFixed(0)}%`);

    if (result.confidence < 0.7) {
      return {
        recipeName: result.recipeName,
        confidence: result.confidence,
        currentStep: "low_confidence",
        exitMessage: `I don't have enough confidence about this recipe: ${result.recipeName} (${(result.confidence * 100).toFixed(0)}%)`,
      };
    }

    return {
      recipeName: result.recipeName,
      cuisine: result.cuisine,
      servings: result.servings,
      confidence: result.confidence,
      currentStep: "recipe_identified",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { errors: [`identifyRecipeNode: ${msg}`], currentStep: "identify_failed" };
  }
}

// ── Node 2: Find Ingredients ───────────────────────────────────────
async function findIngredientsNode(
  state: typeof RecipeState.State
): Promise<typeof RecipeState.Update> {
  console.log("\n┌─ Step 2  Finding ingredients");
  try {
    const structured = model.withStructuredOutput(IngredientsListSchema, { name: "find_ingredients" });
    const result = await structured.invoke([
      new SystemMessage(
        `You are a master chef. List all essential ingredients for a canonical ${state.recipeName}.
Rules:
1. Return base quantities for EXACTLY ONE serving.
2. Ensure all foundational components are included:
   - For baking: leavening agents, fats (butter/oil), binders (eggs/flax), liquids, and flour.
   - For curries: base aromatics, fats, proteins, and spices.
3. Use the following categories ONLY:
   - meat_seafood
   - vegetable_fruit
   - dairy
   - spice_herb
   - grain_flour
   - oil_fat (USE THIS for oils, butter, ghee)
   - sauce_condiment
   - other
4. Be precise with units (grams, ml, pieces, tbsp, tsp, cups).`
      ),
      new HumanMessage(`Recipe: ${state.recipeName}, Cuisine: ${state.cuisine}`),
    ]);

    console.log(`└  Found ${result.ingredients.length} base ingredients`);
    return {
      rawIngredients: result.ingredients,
      currentStep: "ingredients_found",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { errors: [`findIngredientsNode: ${msg}`], currentStep: "find_failed" };
  }
}
// ── Node 3: Validate Ingredients ────────────────────────────────────
async function validateIngredientsNode(
  state: typeof RecipeState.State
): Promise<typeof RecipeState.Update> {
  console.log("\n┌─ Step 3  Validating ingredients & taste profile");
  const ingredientList = state.rawIngredients.map(i => `${i.name} (${i.category})`).join(", ");

  try {
    const structured = model.withStructuredOutput(ValidationSchema, { name: "validate_ingredients" });
    const result = await structured.invoke([
      new SystemMessage(
        `Validate if the ingredients are proper for the recipe and match the user's taste preference.
Rules:
1. isProper: Set to true if the ingredients list is a standard, canonical representation of the dish.
2. matchesTasteProfile:
   - If the user specified a preference (e.g., "extra spicy", "low salt", "very sweet"), check if the ingredients align.
   - If the user did NOT specify any taste preference, set matchesTasteProfile to TRUE by default.
   - Do NOT fail validation just because the user didn't mention a taste.
3. feedback: Provide a concise explanation.`
      ),
      new HumanMessage(
        `User Input: ${state.rawInput}\n` +
        `Recipe: ${state.recipeName}\n` +
        `Ingredients: ${ingredientList}`
      ),
    ]);

    console.log(`│  Is Proper    : ${result.isProper}`);
    console.log(`│  Taste Match  : ${result.matchesTasteProfile}`);
    console.log(`└  Feedback     : ${result.feedback}`);

    if (!result.isProper || !result.matchesTasteProfile) {
      return {
        isProper: result.isProper,
        tasteMatch: result.matchesTasteProfile,
        validationFeedback: result.feedback,
        currentStep: "validation_failed",
        exitMessage: `Ingredients validation failed: ${result.feedback}`,
      };
    }

    return {
      isProper: result.isProper,
      tasteMatch: result.matchesTasteProfile,
      validationFeedback: result.feedback,
      currentStep: "ingredients_validated",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { errors: [`validateIngredientsNode: ${msg}`], currentStep: "validate_failed" };
  }
}

// ── Node 4: Normalize & Calculate Quantities ──────────────────────────
async function normalizeIngredientsNode(
  state: typeof RecipeState.State
): Promise<typeof RecipeState.Update> {
  console.log("\n┌─ Step 4  Calculating quantities & normalizing");
  console.log(`│  Scaling for ${state.servings} servings`);

  const ingredientBlock = state.rawIngredients
    .map((ing, i) => `${i + 1}. ${ing.name} (${ing.quantity} ${ing.unit} base)`)
    .join("\n");

  try {
    const structured = model.withStructuredOutput(NormalizedListSchema, { name: "normalize_ingredients" });
    const result = await structured.invoke([
      new SystemMessage(
        `You are a grocery shopping assistant.
1. Scale all quantities to exactly match the requested servings (${state.servings}).
2. Generate a precise searchQuery for an Indian grocery app.
3. Determine if purchasable (exclude water, salt, basic oil).`
      ),
      new HumanMessage(
        `Recipe: ${state.recipeName}\n` +
        `Servings: ${state.servings}\n\n` +
        `Ingredients (Base for 1 serving):\n${ingredientBlock}`
      ),
    ]);

    console.log(`└  Done ✓`);
    return {
      finalIngredients: result.ingredients,
      currentStep: "done",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { errors: [`normalizeIngredientsNode: ${msg}`], currentStep: "normalize_failed" };
  }
}

// ════════════════════════════════════════════════════════════
//  GRAPH ASSEMBLY
// ════════════════════════════════════════════════════════════

function continueOrStop(state: typeof RecipeState.State) {
  if (state.errors.length > 0 || state.exitMessage) return END;
  return "next";
}

const pipeline = new StateGraph(RecipeState)
  .addNode("detectIntent",         detectIntentNode)
  .addNode("identifyRecipe",       identifyRecipeNode)
  .addNode("findIngredients",       findIngredientsNode)
  .addNode("validateIngredients",  validateIngredientsNode)
  .addNode("normalizeIngredients", normalizeIngredientsNode)

  .addEdge(START, "detectIntent")
  .addConditionalEdges("detectIntent", continueOrStop, {
    next: "identifyRecipe",
    [END]: END,
  })
  .addConditionalEdges("identifyRecipe", continueOrStop, {
    next: "findIngredients",
    [END]: END,
  })
  .addConditionalEdges("findIngredients", continueOrStop, {
    next: "validateIngredients",
    [END]: END,
  })
  .addConditionalEdges("validateIngredients", continueOrStop, {
    next: "normalizeIngredients",
    [END]: END,
  })
  .addEdge("normalizeIngredients", END)
  .compile();

// ════════════════════════════════════════════════════════════
//  OUTPUT DISPLAY
// ════════════════════════════════════════════════════════════

function pad(str: string, len: number): string {
  return str.length >= len ? str.slice(0, len) : str + " ".repeat(len - str.length);
}

function printResults(state: typeof RecipeState.State): void {
  const { recipeName, servings, cuisine, finalIngredients, isProper, tasteMatch, validationFeedback } = state;
  const purchasable = finalIngredients.filter(i => i.purchasable);
  const skipped     = finalIngredients.filter(i => !i.purchasable);

  const DIVIDER = "═".repeat(66);
  const LINE    = "─".repeat(66);

  console.log(`\n${DIVIDER}`);
  console.log(`  ${recipeName}  ·  ${servings} servings  ·  ${cuisine}`);
  console.log(`  Proper: ${isProper ? "✅" : "❌"}  Taste Match: ${tasteMatch ? "✅" : "❌"}`);
  if (validationFeedback) {
    console.log(`  Feedback: ${validationFeedback}`);
  }
  console.log(DIVIDER);

  // ── Items to buy ──────────────────────────────────────────
  console.log(`\n  TO BUY  (${purchasable.length} items)\n`);
  console.log(`  ${pad("INGREDIENT", 26)}  ${pad("QTY + UNIT", 14)}  SEARCH QUERY`);
  console.log(`  ${LINE}`);

  for (const ing of purchasable) {
    const qty  = `${ing.quantity} ${ing.unit}`;
    const flag = ing.optional ? "  (optional)" : "";
    console.log(
      `  ${pad(ing.name, 26)}  ${pad(qty, 14)}  "${ing.searchQuery}"${flag}`
    );
  }

  // ── Items skipped ─────────────────────────────────────────
  if (skipped.length > 0) {
    console.log(`\n  SKIP — assumed already at home  (${skipped.length} items)\n`);
    for (const ing of skipped) {
      console.log(`  •  ${ing.name}`);
    }
  }

  console.log(`\n${DIVIDER}\n`);
}

// ════════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════════

const input = process.argv[2];
if (input === undefined) {
  console.error(
    "Please provide a recipe name and serving count as a command line argument.\n" +
    "Example: npx tsx snapcart-pipeline.ts \"Paneer Butter Masala for 4\""
  );
  process.exit(1);
}

console.log("\n══════════════════════════════════════════════════════════════════");
console.log(`  SnapCart Pipeline`);
console.log(`  Provider : ${CONFIG.provider.toUpperCase()}`);
console.log(`  Model    : ${CONFIG.provider === "gemini" ? CONFIG.gemini.model : CONFIG.ollama.model}`);
console.log(`  Input    : "${input}"`);
console.log("══════════════════════════════════════════════════════════════════");

const result = await pipeline.invoke({
  rawInput: input,
  messages: [new HumanMessage(input)],
});

if (result.errors.length > 0) {
  console.error("\n  PIPELINE ERROR:");
  result.errors.forEach((e: string) => console.error(`  ✗  ${e}`));
  process.exit(1);
}

if (result.exitMessage) {
  console.log(`\n  NOTICE: ${result.exitMessage}\n`);
  process.exit(0);
}

printResults(result);