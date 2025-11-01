from transformers import (
    BlenderbotTokenizer,
    TFBlenderbotForConditionalGeneration,
    pipeline
)


# Model names
DIALOGUE_MODEL = "facebook/blenderbot-400M-distill"
EMOTION_MODEL = "bhadresh-savani/bert-base-uncased-emotion"


print(f"Loading models...\nDialogue: {DIALOGUE_MODEL}\nEmotion: {EMOTION_MODEL}")

# Load Blenderbot tokenizer and TensorFlow model
tokenizer = BlenderbotTokenizer.from_pretrained(DIALOGUE_MODEL)
model = TFBlenderbotForConditionalGeneration.from_pretrained(DIALOGUE_MODEL)  # TensorFlow model


# Load emotion classifier pipeline with PyTorch framework
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
    framework="pt"  # PyTorch model
)

EMPATHY_PROMPT = (
    "You are a gentle, curious, inner-voice AI that guides users by asking reflective questions.\n"
    "You help them think clearly and find strength, not just comfort.\n"
    "Your replies should be short, kind, and introspective — like their inner self speaking.\n\n"
)


def generate_empathic_reply(user_message: str, context: str) -> str:
    # 1. Detect emotion
    emotions = emotion_classifier(user_message)
    detected_emotion = emotions[0]['label'] if emotions else "neutral"

    # 2. Build enhanced context
    enhanced_context = f"{context}\n(User seems to be feeling {detected_emotion.lower()})"

    # 3. Create full prompt
    prompt = (
        EMPATHY_PROMPT
        + f"Recent thoughts: {enhanced_context}\n\n"
        + f"User: {user_message}\n"
        + "Inner Self:"
    )

    # 4. Tokenize and generate response with TensorFlow model
    inputs = tokenizer([prompt], return_tensors="tf", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 5. Extract reply
    reply = text.split("Inner Self:")[-1].strip() if "Inner Self:" in text else text.strip()

    # 6. Cut reply to 400 chars if needed
    return reply[:400]


# Quick test
if __name__ == "__main__":
    user_input = "I’m feeling lonely and far from home."
    conversation_context = "The user mentioned they’re studying away from family."
    response = generate_empathic_reply(user_input, conversation_context)
    print("Inner Self:", response)
