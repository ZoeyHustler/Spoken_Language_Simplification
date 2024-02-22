import openai
API_KEY= "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"
model_id = 'whisper-1'
def Transcribe(audio_file_path):
    #+--------------------------------------------------------------+
    #|Input: string of audio file path                              |
    #|Process: calls openAI Whisper with the audio file and a promptt|
    #|Output: Transcription of the audio file                       |
    #+--------------------------------------------------------------+
    audio_file = open(audio_file_path, "rb")
    Styleprompt = "Hello, welcome to my lecture" #figure out a good prompt to use
    response = openai.Audio.transcribe(api_key = API_KEY,model=model_id,file=audio_file,prompt=Styleprompt, response_format='text')
    return response
