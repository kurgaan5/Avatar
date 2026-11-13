# Источники моделей

Установщик сохраняет файлы в `.runtime/`. Манифест [models.json](../models.json) фиксирует ревизии исходников и SHA-256 проверенных весов. Для собственного кэша можно задать `AVATAR_RUNTIME`. Для уже установленных зависимостей доступны `AVATAR_MODELS`, `AVATAR_DH` и `AVATAR_LIVEPORTRAIT`. Скрипт загрузки раскладывает файлы относительно `AVATAR_RUNTIME`; индивидуальные пути используются при запуске.

| Компонент | Что брать |
| --- | --- |
| Голова и веки | [LivePortrait, код](https://github.com/KlingAIResearch/LivePortrait) и [человеческие веса](https://huggingface.co/KlingTeam/LivePortrait). В этой сборке ориентиры лица берёт MediaPipe |
| Рот | [DH_live Mini](https://github.com/kleinlee/DH_live/tree/4467e97cd97194c2c54762043cf121c6d313db12), браузерный WASM и DINet Mini. Если Google Drive не отдаёт файл напрямую, скачайте `epoch_40_new.pth` из папки автора, указанной в upstream README, и положите в `.runtime/DH_live/checkpoint/DINet_mini/` |
| Ориентиры и силуэт | [MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) и Selfie Multiclass Segmenter |
| Наличие речи | [Silero VAD 5.1.2](https://github.com/snakers4/silero-vad/releases/tag/v5.1.2), CPU ONNX |
| Допуск артикуляции по голосу | [CAM++](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models), `campplus-zh-en-advanced.onnx`, CPU ONNX |
| Клон голоса | [Qwen3-TTS 12Hz 1.7B Base](https://github.com/QwenLM/Qwen3-TTS), [GGUF Q8_0](https://huggingface.co/CC-TM/Qwen3-TTS-GGUF), Python-обёртка `realtimetts-qwen-native` |
| Распознавание | [Whisper.cpp](https://github.com/ggml-org/whisper.cpp), [large-v3-turbo Q5_0](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo-q5_0.bin) |
| Текст ответов | [llama.cpp](https://github.com/ggml-org/llama.cpp), [Qwen3.5 9B GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) или [Qwen3.5 4B GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF). Выберите `Q4_K_M` и сохраните его ревизию для повторения своего прогона |
| Новый персонаж | [NVIDIA Audio2Face-3D SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK) и совместимые веса по инструкции NVIDIA. Используется при реализации 3D-режима |

Весовые файлы, готовые профили, драйверы, сторонние SDK и записи встреч хранятся отдельно. На сторонние компоненты распространяются их собственные условия. Например, upstream LivePortrait отдельно ограничивает модели детекции InsightFace. Используйте собственные записи либо материалы с разрешением владельца.
