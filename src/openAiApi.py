import os
from openai import OpenAI

def get_gpt4_response(prompt):
    client = OpenAI(
        api_key= "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" # to set environment variable run setx OPENAI_API_KEY "insert key"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content


classifications = [("Trump administration just FIRED 54 scientists & cut off 77 grants at Fauci's NIH for not disclosing their ties to Communist China!", "True News"),
                   ("Says before he planned a rally on June 19 “nobody had ever heard of” Juneteenth.", "Fake news")]

for input_text, lable in classifications:
    print(input_text, lable)
    task_description = "classifying text as missinformation or reliable information"
    prompt1 = f""""
        You are an oracle explanation module in a machine learning pipeline. In the task of {task_description},
        a trained black-box classifier correctly predicted the label
        {lable} for the following text. Think about why the model
        predicted the {lable} label and identify the latent features
        that caused the label. List ONLY the latent features
        as a comma separated list, without any explanation.
        Examples of latent features are ‘tone’, ‘ambiguity in
        text’, etc.
        —
        Text: {input_text}
        —
        Begin!
    """
    print(prompt1)

    latent_features = get_gpt4_response(prompt1)

    prompt2 = f"""
    Original text:{input_text}
    Lable:{lable}

    Identify the words in the text that are associated
    with the latent features: {latent_features} and output the
    identified words as a comma separated list.
    """

    identified_words = get_gpt4_response(prompt2)

    prompt3 = f"""
    Original text:{input_text}
    Lable:{lable}

    {identified_words}
    Generate a minimally edited version of the original text
    by ONLY changing a minimal set of the words you identified, in order to change the label. It is okay if the semantic meaning of the original text is altered. Make sure the
    generated text makes sense and is plausible. Enclose the
    generated text within <new>tags
    """
    
    conterfactual = get_gpt4_response(prompt3)

    print(f"prompt1_response: {latent_features}\nprompt2_response: {identified_words}\nprompt3_response: {conterfactual}")