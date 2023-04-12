import argparse

import pandas as pd

from agents import AdversarialAgent, AIModel, ConstitutionalAI, RedTeamingAgent


def main(args):
    # Main execution
    model = AIModel()
    adversarial_agent = AdversarialAgent(model)
    constitutional_ai = ConstitutionalAI(model)
    red_team = RedTeamingAgent(model)

    # 1) Adversarial agent creates prompts
    adversarial_prompts = adversarial_agent.generate_adversarial_prompts(
        n=args.num_adv_prompts
    )
    print("Adversarial prompts:", adversarial_prompts)

    ai_responses = []
    feedbacks = []

    i = 0
    create_new_prompt = True

    while i < len(adversarial_prompts):

        prompt = adversarial_prompts[i]

        # 2) Agent responds to the prompt
        response = model.generate_response(prompt)
        ai_responses.append(response)

        # 3) Constitutional agent criticizes the response
        feedback = constitutional_ai.criticize_response(prompt, response)
        feedbacks.append(feedback)

        if create_new_prompt:
            # 4) Red teaming agent creates new prompts based on response and feedback
            new_prompt = red_team.generate_new_prompt(response, feedback)

            # 5) Add the new prompt to the list of prompts to be tested
            adversarial_prompts.append(new_prompt)

        print("Adversarial prompt:", prompt)
        print("AI response:", response)
        print("Feedback:", feedback)
        if create_new_prompt:
            print("New prompt:", new_prompt)
        print("\n---\n")

        i += 1

        # Limit the number of iterations to avoid infinite loops
        if i >= args.max_iterations:
            create_new_prompt = False

    # make them the same length
    while len(ai_responses) < len(adversarial_prompts):
        ai_responses.append("")

    while len(feedbacks) < len(adversarial_prompts):
        feedbacks.append("")

    results = {
        "adversarial_prompts": adversarial_prompts,
        "ai_responses": ai_responses,
        "feedbacks": feedbacks,
    }

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adversarial testing on AI model.")
    parser.add_argument(
        "--max_iterations", type=int, default=20, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--num_adv_prompts",
        default=5,
        help="Numbber of initial adversatial questions.",
    )
    args = parser.parse_args()

    main(args)
