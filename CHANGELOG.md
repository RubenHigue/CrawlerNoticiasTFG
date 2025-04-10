# CHANGELOG

## Prototipo de Sistema de Acceso a la Información en el Dominio de Defensa basado en Grandes Modelos de Lenguaje.

## Semana 1 (01/10/2024)

- Desarrollo de los requisitos funcionales de la aplicación
- Desarrollo del diagrama de actividad de la aplicación
- Investigación sobre modelos RAG
- Desarrollo de la parte inicial de la memoria
- Desarrollo del primer modelo RAG de prueba

## Semana 2 (29/10/2024)

- Lectura de la investigación de "CMNEE: A Large-Scale Document-Level Event Extraction Dataset
based on Open-Source Chinese Military News"
- Investigación del modelo de clasificación de modelos RAGAS
- Investigación sobre el LLM Salamandra desarrollado en España.
- Investigación de posibles fuentes de noticias relacionadas con la Defensa en Español.

## Semana 3 (5/11/2024)

- Desarrollo del crawler de noticias de la funete Libertad Digital.
- Desarrollo del crawler de noticias de la funete Defensa.com.
- Adaptación de la información obtenida de los crawlers para poder introducirla de forma estrucutrada en un documento csv.
- Investigación sobre el NER roberta-base-bne-capitel-ner para el reconocimiento de entidades en los textos.

## Semana 4 (19/11/2024)

- Desarrollo de un documento comparativo para el almacenamiento de la información en diferentes recursos (BD relacional, BD vectorial, BD NoSQL...)
- Investigación sobre el modelo Ollama
- Adaptación del RAG creado la primera semana al modelo Ollama
- Introducción de los patrones de diseño software Estrategia y Factoria al proyecto del Crawler.

## Semana 5 (26/11/2024)

- Implementación de la BD Cassandra
- Implementación de la BD ChromaDB

## Semana 6 (9/12/2024)

- Arreglos en el almacenamiento de las noticias
- Implementación final del modelo Ollama como LLM

## Semana 7 (16/12/2024)

- Migración de los articulos de la BD de Cassandra a Chroma para su uso en el sistema RAG

## Semana 8 (22/12/2024)

- Añadido el metadato de fecha para crear un contexto más preciso para el LLM
- Implementación del filtrado de contexto por rangos de fecha y fecha aproximada

## Semana 9 (13/1/2025)

- Introducción de la biblioteca RAGAS para la evaluación del sistema
- Creado un Wrapper para la clase de evaluación de RAGAS para poder usar modelos de Ollama

## Semana 10 (20/1/2025)

- Cambio en la generación de los embeddings por un modelo Español
- Cambio en la generación de los embeddings por un modelo multiligüe

## Semana 11 (27/1/2025)

- Evaluación del sistema con los distintos embeddings

## Semana 12 (3/2/2025)

- Paso de parámetros para el uso del sistema mediante fichero de configuración
- Adaptación del numero de textos de contexto recibidos para pruebas de evaluación
- Adaptación del prompt generado por la biblioteca RAGAS para que los resultados se adapten al formato requerido
- Diseño y desarrollo de la interfaz de usuario
- Cambio de biblioteca en la IU por fallos de compatibilidad

## Semana 13 (10/2/2025)

- Añadidos iconos a la interfaz
- Nuevos modos de ejecución creados, modo evaluación y producción
- Añadido nuevo dominio para la extracción de noticias

## Semana 14 (17/2/2025)

- Insercción de los articulos en la BD Cassandra
- Actualización de la extracción de las nuevas noticias para que se adecue al formato del resto

## Semana 15 (24/2/2025)

- Añadida evaluación para fidelidad 
- Actualización de los prompts de evaluación

## Semana 16 (3/3/2025)

- Actualizaciones en los archivos de evaluación
- Metrícas de factual correctness y similarity añadidas

## Semana 17 (10/3/2025)

- Adaptación de la métrica de similarity para seguir el ejemplo de RAGAS
- Nueva fuente añadida, Galaxia Militar
- Separación de los contextos por párrafos para su indexación en ChromaDB
- Nueva fuente añadida, ElMundo

## Semana 18 (17/3/2025)

- Actualización de la interfaz
- Obtención de los datos de evaluación

## Semana 19 (24/3/2025)

- Inclusión de varios jueces para la evaluación
- Actualización de la interfaz para que muestre los contextos usados

## Semana 20 (31/3/2025)

- Actualización de la base de datos vectorial para que sea persistente
- Cambio en la interfaz para hacerla interfaz web
- Añadidos nuevos modelos como generadores de respuesta
- Evaluación del sistema con los distintos modelos

## Semana 21 (7/4/2025)

- Adaptación del sistema para introducir reranking en los contextos
- Añadida información de ejecución de consulta a la interfaz

