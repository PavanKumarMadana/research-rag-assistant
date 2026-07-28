"""
LLM Service Module.

Provides unified interface for LLM providers (Gemini, OpenAI).
"""

import time
from typing import Optional
from enum import Enum

from loguru import logger

from backend.app.core.config import settings


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"


class LLMService:
    """Service for interacting with Large Language Models."""

    def __init__(self) -> None:
        """Initialize the LLM service."""
        self.provider = settings.LLM_PROVIDER
        self._gemini_model = None
        self._openai_client = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the appropriate LLM client based on configuration."""
        if self.provider == LLMProvider.GEMINI:
            self._init_gemini()
        elif self.provider == LLMProvider.OPENAI:
            self._init_openai()
        else:
            logger.warning(f"Unknown LLM provider: {self.provider}. Falling back to Gemini.")
            self.provider = LLMProvider.GEMINI
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai

            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("GEMINI_API_KEY not set. Some LLM features will not work.")

            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                settings.LLM_MODEL,
                generation_config={
                    "temperature": settings.TEMPERATURE,
                    "max_output_tokens": settings.MAX_TOKENS,
                },
            )
            logger.info(f"Gemini initialized with model: {settings.LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._gemini_model = None

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = settings.OPENAI_API_KEY
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. Some LLM features will not work.")

            self._openai_client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI initialized with model: {settings.OPENAI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self._openai_client = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instructions.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            str: Generated text.
        """
        if self.provider == LLMProvider.GEMINI:
            return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Gemini.

        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            str: Generated text.
        """
        try:
            if not self._gemini_model:
                return "LLM is not configured. Please set GEMINI_API_KEY in your environment."

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self._gemini_model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature or settings.TEMPERATURE,
                    "max_output_tokens": max_tokens or settings.MAX_TOKENS,
                },
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using OpenAI.

        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            str: Generated text.
        """
        try:
            if not self._openai_client:
                return "LLM is not configured. Please set OPENAI_API_KEY in your environment."

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=temperature or settings.TEMPERATURE,
                max_tokens=max_tokens or settings.MAX_TOKENS,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def is_available(self) -> bool:
        """Check if LLM service is available.

        Returns:
            bool: True if LLM is configured and available.
        """
        if self.provider == LLMProvider.GEMINI:
            return self._gemini_model is not None and bool(settings.GEMINI_API_KEY)
        else:
            return self._openai_client is not None and bool(settings.OPENAI_API_KEY)