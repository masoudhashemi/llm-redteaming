This diagram outlines the main steps in the flow:

1. The `AdversarialAgent` generates a set of prompts to be fed to the AI model.
2. The AI model (`AIModel`) responds to each adversarial prompt.
3. The `ConstitutionalAI` criticizes the response and provides feedback.
4. The `RedTeamingAgent` uses the feedback and response to create new prompts to challenge the AI further.