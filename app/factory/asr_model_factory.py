from app.config import CONFIG


class ASRModelFactory:
    @staticmethod
    def create_asr_model():
        if CONFIG.ASR_ENGINE == "faster_whisper":
            from app.asr_models.faster_whisper_engine import FasterWhisperASR
            return FasterWhisperASR()
        else:
            raise ValueError(f"Unsupported ASR engine: {CONFIG.ASR_ENGINE}")
