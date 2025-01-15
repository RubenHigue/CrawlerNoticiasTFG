from abc import ABC

import typing as t

from ragas.exceptions import LLMDidNotFinishException
from ragas.llms import BaseRagasLLM
from langchain.schema import PromptValue, LLMResult, Generation
import ollama
from ragas.run_config import add_async_retry
from ragas.run_config import RunConfig
import asyncio
import json


def parse_json_output(output: str):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("La salida no es un JSON válido: " + output)


class BaseLLMOllama(BaseRagasLLM, ABC):
    def __init__(self, model_name: str, run_config: RunConfig):
        """
        Inicializa el modelo de Ollama.
        :param model_name: Nombre del modelo en Ollama (por ejemplo, 'llama3.2').
        :param run_config: Configuración de reintentos y otros parámetros.
        """
        self.model_name = model_name
        self.run_config = run_config

    async def agenerate_text(
            self,
            prompt: str,
            n: int = 1,
            temperature: t.Optional[float] = None,
            stop: t.Optional[t.List[str]] = None,
            callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """
        Genera texto de manera asincrónica.
        :param prompt: Texto del prompt.
        :param n: Número de respuestas a generar.
        :param temperature: Controla la aleatoriedad en la salida.
        :param stop: Tokens para detener la generación (no soportado por Ollama).
        :param callbacks: Callbacks opcionales.
        :return: Respuesta generada.
        """
        from asyncio import to_thread

        # Ollama no soporta directamente múltiples generaciones, se hace n veces
        generations = []
        prompt = prompt + ('Rellena el apartado Output de la misma forma que en el ejemplo de Albert Einstein pero '
                           'para "la answer dada, 1 si tiene relación con el artículo y 0 si no tiene nada de '
                           'relación. Dado el input, dada la "question" procesa el "context" y saca solo el formato '
                           'JSON con los siguientes campos: "statement", el apartado answer del input. "reason", '
                           'el trozo de contexto por el que se ha guiado la respuesta y "attributed", que sera la '
                           'puntuación o 1 o 0 en cuestión de la relevancia. Por favor devuelve el Output para que se '
                           'adecue al siguiente JSON Schema rellenando los datos de statement, reason y attributed: '
                           '"classifications": [{"statement": "Albert Einstein, born on 14 March 1879, '
                           'was a German-born theoretical physicist, widely held to be one of the greatest and most '
                           'influential scientists of all time.","reason": "The date of birth of Einstein is '
                           'mentioned clearly in the context.","attributed": 1},')
        for _ in range(n):
            response = await to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            message = response.get('message', "").get('content')
            # message = parse_json_output(message)
            generations.append(Generation(text=message,
                                          generation_info=None, ))

        return LLMResult(generations=[generations])

    def generate_text(
            self,
            prompt: str,
            n: int = 1,
            temperature: t.Optional[float] = None,
            stop: t.Optional[t.List[str]] = None,
            callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """
        Genera texto de manera sincrónica.
        :param prompt: Texto del prompt.
        :param n: Número de respuestas a generar.
        :param temperature: Controla la aleatoriedad en la salida.
        :param stop: Tokens para detener la generación.
        :param callbacks: Callbacks opcionales.
        :return: Respuesta generada.
        """
        return asyncio.run(
            self.agenerate_text(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        )

    async def generate(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: t.Optional[float] = None,
            stop: t.Optional[t.List[str]] = None,
            callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """
        Genera texto utilizando el bucle de eventos asincrónico.
        :param prompt: Instancia de PromptValue con el texto del prompt.
        :param n: Número de respuestas a generar.
        :param temperature: Controla la aleatoriedad en la salida.
        :param stop: Tokens para detener la generación.
        :param callbacks: Callbacks opcionales.
        :return: Instancia de LLMResult con las respuestas generadas.
        """
        if temperature is None:
            temperature = self.get_temperature(n)

        agenerate_text_with_retry = add_async_retry(
            self.agenerate_text, self.run_config
        )

        result = await agenerate_text_with_retry(
            prompt=prompt.to_string(),
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        print("HOLA")

        return result
