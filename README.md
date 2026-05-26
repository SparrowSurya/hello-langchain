# hello-langchain

To install dependencies:

```bash
bun install
```

To run:

```bash
bun start
```

This project was created using `bun init` in bun v1.3.14. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.

```mermaid
flowchart TD
    START(("START"))
    FINISH(("FINISH"))
    CANCEL(("CANCEL"))

    textInput[/"User Input
    User types recipe name, description, or image caption"/]

    classifyIntent["Classify Intent
    LLM — determines if input is recipe-related"]

    suggestRecipe[/"Clarify Intent
    User picks from suggestions or rephrases input"/]

    detectRecipe["Detect Recipe
    LLM — extracts recipe name from input"]

    confirmRecipe[/"Confirm Recipe
    User confirms, corrects, or cancels the detected recipe"/]

    manualEntry[/"Re-enter Recipe
    User types the correct recipe name manually"/]

    extractIngredients["Extract Ingredients
    LLM — generates full ingredient list for the recipe"]

    reviewIngredients[/"Review Ingredients
    User approves, replaces, or skips individual items"/]

    scaleQuantity["Scale Quantities
    LLM — scales ingredient amounts to serving count"]

    askServings[/"Enter Serving Size
    User specifies the number of servings required"/]

    mapProducts["Map to Instamart Products
    Search API — finds best matching product per ingredient"]

    resolveMissing[/"Resolve Unavailable Items
    User picks an alternative product or removes the item"/]

    confirmBrand[/"Select Brand Preference
    User picks a preferred brand or accepts the default match"/]

    reviewCart[/"Review Cart
    User sees full item list, quantities, and total cost"/]

    intentDec{"Input type?"}
    confidenceDec{"Detection
    confidence?"}
    recipeCorrectDec{"Recipe
    correct?"}
    retryDec{"Retry
    attempts
    under 3?"}
    servingsDec{"Servings
    found in
    input?"}
    missingDec{"All items
    available?"}
    brandDec{"Multiple
    brands
    available?"}
    cartDec{"User
    action?"}

    START --> textInput
    textInput --> classifyIntent
    classifyIntent --> intentDec

    intentDec -->|"clear recipe request"| detectRecipe
    intentDec -->|"vague or unclear"| suggestRecipe
    intentDec -->|"off-topic — not a recipe"| CANCEL

    suggestRecipe -->|"user picks a recipe"| detectRecipe
    suggestRecipe -->|"user cancels"| CANCEL

    detectRecipe --> confidenceDec

    confidenceDec -->|"high — 0.85 or above
    auto-proceed"| extractIngredients
    confidenceDec -->|"low — below 0.85
    ask user to confirm"| confirmRecipe

    confirmRecipe --> recipeCorrectDec
    recipeCorrectDec -->|"confirmed correct"| extractIngredients
    recipeCorrectDec -->|"wrong — let me fix it"| manualEntry
    recipeCorrectDec -->|"cancel"| CANCEL

    manualEntry --> retryDec
    retryDec -->|"yes — retry"| detectRecipe
    retryDec -->|"no — max 3 attempts reached"| CANCEL

    extractIngredients --> reviewIngredients
    reviewIngredients -->|"confirmed"| scaleQuantity
    reviewIngredients -->|"cancel"| CANCEL

    scaleQuantity --> servingsDec
    servingsDec -->|"yes — already known"| mapProducts
    servingsDec -->|"no — not mentioned"| askServings
    askServings --> mapProducts

    mapProducts --> missingDec
    missingDec -->|"all items found"| brandDec
    missingDec -->|"one or more unavailable"| resolveMissing
    resolveMissing -->|"resolved"| brandDec
    resolveMissing -->|"cancel"| CANCEL

    brandDec -->|"yes — show brand options"| confirmBrand
    brandDec -->|"no — single option only"| reviewCart
    confirmBrand --> reviewCart

    reviewCart --> cartDec
    cartDec -->|"confirm — add to cart"| FINISH
    cartDec -->|"edit ingredients"| reviewIngredients
    cartDec -->|"cancel"| CANCEL
```

THE EXTRA DIMENSIONS YOU NEED TO CAPTURE
Beyond just the recipe name, your pipeline needs to resolve five dimensions before ingredient extraction makes sense.
DIMENSION 1 — Variant
Which specific version of this recipe does the user want. For dosa: plain, masala, rava, Mysore masala, etc. This is the most important dimension because it determines the core ingredient list.
DIMENSION 2 — Experience Level
Is this person cooking this dish for the first time or have they made it before. A first-time dosa maker needs simpler instructions and may benefit from the rava variant over the fermented batter variant simply because it is achievable the same day.
DIMENSION 3 — Time Availability
Do they want to cook this today or are they planning ahead. Fermented dosa batter needs overnight prep. If the user says "I want to make dosa tonight", the LLM should flag this and either suggest rava dosa or ask if they already have fermented batter ready.
DIMENSION 4 — Accompaniments
Does the user want ingredients only for the dosa itself, or also for sambar, coconut chutney, or other traditional sides. This can double the ingredient list.
DIMENSION 5 — Dietary and Substitution Constraints
Vegan users cannot use ghee. Some users may be allergic to urad dal. A North Indian unfamiliar with South Indian cooking may not be able to find certain ingredients locally and may need substitutions.

* recipeName: Dosa
* variant: null — must resolve before proceeding
* filling: null — must resolve before proceeding
* accompaniments: [] — optional, defaults to none
* timeConstraint: tonight — inferred from user input
* fermentationPossible: false — inferred from timeConstraint
* recommendedVariant: Rava Dosa — system suggestion based on timeConstraint
* servings: null — must resolve before proceeding
* dietaryConstraints: [] — defaults to none
* substituteNeeded: false
