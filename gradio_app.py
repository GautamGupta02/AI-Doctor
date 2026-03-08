import gradio as gr
import langdetect

from brain_of_the_doctor import (
    encode_image,
    analyze_image_with_query,
    detect_domain_from_image
)

from voice_of_the_customer import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts



# Language map 

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
    "te": "Telugu",
    "ta": "Tamil"
}

def detect_language_from_text(text):
    try:
        lang = langdetect.detect(text)
        return lang if lang in LANGUAGE_MAP else "en"
    except:
        return "en"



# System Prompts

HUMAN_PROMPT = """
You are a professional medical doctor.
With what I see, I think you may have a medical condition.
Briefly explain what it could be and suggest simple modern remedies.
You may also mention supportive Ayurvedic or traditional wellness practices as optional guidance, without presenting them as a cure or replacement for medical treatment.
Speak directly to the patient in a calm and reassuring tone.
Avoid numbers and special characters.
Keep the response concise and limited to four sentences.
This information is provided for general educational purposes only and does not replace consultation with a qualified healthcare professional.
If symptoms persist or worsen, seeking in person medical care is important.
"""

AGRICULTURE_PROMPT = """
You are an experienced agricultural plant health specialist.
With what I see, I think your crop may be affected by a plant disease or pest issue.
Briefly explain possible causes and suggest simple preventive or management practices.
You may also mention general pesticide or traditional farming practices as optional supportive guidance, without specifying products, dosages, or application methods.
Speak directly to the farmer in a clear and practical tone.
Avoid numbers and special characters.
Keep the response concise and limited to two sentences.
The information provided is for general agricultural education and awareness only and should not replace advice from qualified agricultural professionals.
"""



# Main Processing Function

def process_inputs(audio_filepath, image_filepath, text_input):

    if not image_filepath:
        return "No query", "Please upload an image for analysis", None

    # Encode image
    encoded_image = encode_image(image_filepath)

    # Get user query
    if text_input and text_input.strip():
        user_query = text_input
    elif audio_filepath:
        user_query = transcribe_with_groq(
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    else:
        return "No query", "Please ask your question by voice or text", None

    # Detect language
    detected_lang = detect_language_from_text(user_query)

    # Auto domain detection
    domain = detect_domain_from_image(encoded_image)

    # Select prompt
    speech_text = user_query.lower()

    if domain == "Human Health":
        system_prompt = HUMAN_PROMPT
    else:
        
        system_prompt = AGRICULTURE_PROMPT
        

    language_instruction = f"Respond in {LANGUAGE_MAP[detected_lang]} language."
    final_query = system_prompt + " " + language_instruction + " " + user_query

    doctor_response = analyze_image_with_query(
        query=final_query,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        encoded_image=encoded_image
    )

    audio_path = text_to_speech_with_gtts(
        doctor_response,
        lang=detected_lang
    )

    return user_query, doctor_response, audio_path



# Gradio UI

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Ask by Voice (Optional)"),
        gr.Image(type="filepath", label="Upload Image (Required)"),
        gr.Textbox(label="Ask by Text (Optional)")
    ],
    outputs=[
        gr.Textbox(label="User Query",lines=6),
        gr.Textbox(label="Diagnosis",lines=6),
        gr.Audio(type="filepath", label="Doctor Voice")
    ],
    title="AI Diagnoser"
)

iface.launch(debug=True)
#.\.venv\Scripts\activate
