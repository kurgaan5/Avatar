"""Local audio question → Whisper.cpp → llama.cpp → Qwen3-TTS WAV."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import time
from urllib.request import Request, urlopen

from avatar_paths import MODELS
from avatar_voice import Voice, decode_audio, open_engine, write_wave


def transcribe(whisper, model, question):
    with TemporaryDirectory() as folder:
        audio = Path(folder) / 'question.wav'
        write_wave(audio, decode_audio(question, 16000), 16000)
        command = [whisper, '-m', str(model), '-f', str(audio), '-l', 'ru',
                   '-nt', '-otxt', '-of', str(audio.with_suffix(''))]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return audio.with_suffix('.txt').read_text(encoding='utf-8').strip()


def answer(url, model, persona, question):
    payload = {
        'model': model, 'temperature': .25, 'max_tokens': 100,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'system', 'content': persona},
                     {'role': 'user', 'content': question}],
    }
    request = Request(url.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=45) as response:
        return json.load(response)['choices'][0]['message']['content'].strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--persona', type=Path)
    parser.add_argument('--whisper', default='whisper-cli')
    parser.add_argument('--whisper-model', type=Path,
                        default=MODELS/'asr/ggml-large-v3-turbo-q5_0.bin')
    parser.add_argument('--model', default='avatar-local')
    parser.add_argument('--url', default='http://127.0.0.1:18083/v1')
    args = parser.parse_args()
    persona = ('Отвечай естественно и кратко на русском. Если не знаешь факт, скажи об этом. '
               'Произноси одну или две короткие фразы.')
    if args.persona:
        persona += '\n' + args.persona.read_text(encoding='utf-8')
    with open_engine() as engine:
        voice = Voice(engine, args.reference)
        start = time.perf_counter()
        question = transcribe(args.whisper, args.whisper_model, args.question)
        if not question:
            raise SystemExit('Whisper did not detect speech.')
        heard = time.perf_counter()
        reply = answer(args.url, args.model, persona, question)
        answered = time.perf_counter()
        voice.render(reply, args.output)
        print(json.dumps({'answer': reply, 'asr_s': heard-start,
                          'llm_s': answered-heard, 'tts_s': time.perf_counter()-answered},
                         ensure_ascii=False))
