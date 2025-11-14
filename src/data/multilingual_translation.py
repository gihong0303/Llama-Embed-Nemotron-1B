"""
Multilingual Translation Pipeline

Supports 250+ languages as in MMTEB benchmark.
Creates cross-lingual training data through translation.

Based on paper approach: Translate high-quality datasets to multiple languages.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List, Dict, Optional
import json
from tqdm import tqdm
from pathlib import Path


class MultilingualTranslator:
    """
    Translate texts to multiple languages for cross-lingual training.

    Supports:
    - M2M100 (Facebook): 100 languages
    - NLLB (Meta): 200+ languages
    - Custom translation models

    Args:
        model_name: Translation model name
        device: Device to run on
    """

    # Language codes for M2M100 (100 languages)
    M2M100_LANGS = [
        "en", "zh", "es", "ar", "fr", "de", "ja", "ko", "pt", "ru",
        "it", "nl", "pl", "tr", "vi", "id", "th", "uk", "ro", "cs",
        "hi", "he", "fa", "hu", "fi", "sv", "no", "da", "el", "bg",
        # ... (100 total)
    ]

    # Common languages for focused translation
    COMMON_LANGS = ["en", "zh", "es", "ar", "fr", "de", "ja", "ko", "pt", "ru", "hi"]

    def __init__(
        self,
        model_name: str = "facebook/m2m100_418M",
        device: Optional[str] = None,
        use_nllb: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.use_nllb = use_nllb

        print(f"Loading translation model: {model_name}")

        if use_nllb or "nllb" in model_name.lower():
            # NLLB-200 (Meta, 200+ languages)
            from transformers import NllbTokenizer
            self.tokenizer = NllbTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif "m2m100" in model_name.lower():
            # M2M100 (Facebook, 100 languages)
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        else:
            # Generic seq2seq model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

        print(f"Translation model loaded on {device}")

    def translate(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "es",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[str]:
        """
        Translate texts from source to target language.

        Args:
            texts: List of texts to translate
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "es")
            batch_size: Batch size for translation
            max_length: Max sequence length

        Returns:
            translations: List of translated texts
        """
        all_translations = []

        # Set source language
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = source_lang

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            # Generate translations
            with torch.no_grad():
                if hasattr(self.model, "generate"):
                    # Set target language for generation
                    if hasattr(self.tokenizer, "get_lang_id"):
                        forced_bos_token_id = self.tokenizer.get_lang_id(target_lang)
                    else:
                        forced_bos_token_id = None

                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=max_length,
                        num_beams=5,
                        early_stopping=True,
                    )
                else:
                    outputs = self.model(**inputs).logits.argmax(dim=-1)

            # Decode
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_translations.extend(translations)

        return all_translations

    def translate_dataset(
        self,
        input_file: str,
        output_file: str,
        source_lang: str = "en",
        target_langs: List[str] = None,
        fields_to_translate: List[str] = ["query", "positive"],
        batch_size: int = 16,
        show_progress: bool = True,
    ):
        """
        Translate entire dataset to multiple languages.

        Args:
            input_file: Input JSONL file
            output_file: Output JSONL file (will contain all translations)
            source_lang: Source language
            target_langs: Target languages (default: common languages)
            fields_to_translate: Which fields to translate
            batch_size: Translation batch size
            show_progress: Show progress bar
        """
        if target_langs is None:
            target_langs = self.COMMON_LANGS

        print(f"Translating dataset from {source_lang} to {len(target_langs)} languages...")
        print(f"  Target languages: {', '.join(target_langs)}")
        print(f"  Fields: {', '.join(fields_to_translate)}")

        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        print(f"  Loaded {len(data):,} samples")

        all_translations = []

        # Translate to each target language
        for target_lang in target_langs:
            if target_lang == source_lang:
                # Skip source language
                continue

            print(f"\nTranslating to {target_lang}...")

            # Collect texts to translate
            texts_to_translate = {field: [] for field in fields_to_translate}

            for sample in data:
                for field in fields_to_translate:
                    if field in sample:
                        texts_to_translate[field].append(sample[field])

            # Translate each field
            translations = {}
            for field, texts in texts_to_translate.items():
                if show_progress:
                    iterator = tqdm(
                        range(0, len(texts), batch_size),
                        desc=f"  {field}",
                    )
                else:
                    iterator = range(0, len(texts), batch_size)

                field_translations = []
                for i in iterator:
                    batch = texts[i:i + batch_size]
                    batch_translations = self.translate(
                        batch,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        batch_size=len(batch),
                    )
                    field_translations.extend(batch_translations)

                translations[field] = field_translations

            # Create translated samples
            for i, sample in enumerate(data):
                translated_sample = sample.copy()

                # Update translated fields
                for field in fields_to_translate:
                    if field in sample:
                        translated_sample[field] = translations[field][i]

                # Add language metadata
                translated_sample["source_lang"] = source_lang
                translated_sample["target_lang"] = target_lang
                translated_sample["is_translated"] = True

                all_translations.append(translated_sample)

        # Save all translations
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_translations:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\nTranslation complete!")
        print(f"  Input samples: {len(data):,}")
        print(f"  Output samples: {len(all_translations):,}")
        print(f"  Languages: {len(target_langs) - (1 if source_lang in target_langs else 0)}")
        print(f"  Saved to: {output_file}")


class CrossLingualDatasetBuilder:
    """
    Build cross-lingual training data for multilingual embedding models.

    Strategies:
    1. Translate high-quality English datasets to multiple languages
    2. Create cross-lingual pairs (e.g., English query → Spanish document)
    3. Mix monolingual and cross-lingual data
    """

    def __init__(self, translator: MultilingualTranslator):
        self.translator = translator

    def create_crosslingual_pairs(
        self,
        input_file: str,
        output_file: str,
        source_lang: str = "en",
        target_langs: List[str] = None,
        crosslingual_ratio: float = 0.3,
    ):
        """
        Create cross-lingual query-document pairs.

        Example:
        - Original: English query → English document
        - Cross-lingual: English query → Spanish document

        Args:
            input_file: Input JSONL (source language)
            output_file: Output JSONL (mixed mono + cross-lingual)
            source_lang: Source language
            target_langs: Target languages
            crosslingual_ratio: Fraction of cross-lingual pairs
        """
        if target_langs is None:
            target_langs = ["es", "zh", "fr", "de", "ar"]

        print(f"Creating cross-lingual pairs...")
        print(f"  Cross-lingual ratio: {crosslingual_ratio:.1%}")

        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        num_crosslingual = int(len(data) * crosslingual_ratio)
        num_monolingual = len(data) - num_crosslingual

        print(f"  Monolingual: {num_monolingual:,} samples")
        print(f"  Cross-lingual: {num_crosslingual:,} samples")

        # Keep monolingual samples
        monolingual_samples = data[:num_monolingual]

        # Create cross-lingual samples
        crosslingual_samples = []

        for i, sample in enumerate(tqdm(data[num_monolingual:], desc="Creating cross-lingual")):
            # Choose random target language
            import random
            target_lang = random.choice(target_langs)

            # Translate positive document to target language
            positive_translated = self.translator.translate(
                [sample['positive']],
                source_lang=source_lang,
                target_lang=target_lang,
            )[0]

            # Translate negatives (if present)
            negatives_translated = []
            if 'negatives' in sample and sample['negatives']:
                negatives_translated = self.translator.translate(
                    sample['negatives'],
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

            # Create cross-lingual sample
            crosslingual_sample = {
                "query": sample['query'],  # Keep query in source language
                "positive": positive_translated,  # Translate document
                "negatives": negatives_translated,
                "query_lang": source_lang,
                "doc_lang": target_lang,
                "is_crosslingual": True,
                "task_type": sample.get("task_type", "retrieval"),
                "instruction": sample.get("instruction", ""),
            }

            crosslingual_samples.append(crosslingual_sample)

        # Combine
        all_samples = monolingual_samples + crosslingual_samples

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Cross-lingual dataset created: {output_file}")
        print(f"  Total: {len(all_samples):,} samples")


# Example usage
if __name__ == "__main__":
    # Create translator
    translator = MultilingualTranslator(
        model_name="facebook/m2m100_418M",  # or "facebook/nllb-200-distilled-600M"
        device="cuda",
    )

    # Translate dataset
    translator.translate_dataset(
        input_file="data/en_retrieval.jsonl",
        output_file="data/multilingual_retrieval.jsonl",
        source_lang="en",
        target_langs=["es", "zh", "fr", "de", "ja", "ko", "ar", "ru"],
        fields_to_translate=["query", "positive"],
    )

    # Create cross-lingual pairs
    builder = CrossLingualDatasetBuilder(translator)
    builder.create_crosslingual_pairs(
        input_file="data/en_retrieval.jsonl",
        output_file="data/crosslingual_retrieval.jsonl",
        source_lang="en",
        target_langs=["es", "zh", "fr", "de"],
        crosslingual_ratio=0.3,
    )
