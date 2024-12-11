# Instalaciones y prompts
from transformers import pipeline, set_seed
import os


# Salamandra
model_id = "BSC-LT/salamandra-7b"


# Crear el pipeline
generator = pipeline(
    "text-generation",
    model_id,
    device_map="auto"
)


# Argumentos del pipeline
generation_args = {
  "temperature": 0.1,
  "top_p": 0.8,
  "max_new_tokens": 500,
  "repetition_penalty": 1.2,
  "do_sample": True,
  "pad_token_id": generator.tokenizer.eos_token_id
}
set_seed(1)


# Textos para los prompts
intro = "Transformaremos un texto aplicando la siguiente pauta de ortotipografía. No debe perderse nada del significado ni las ideas, debemos modificar lo mínimo posible el texto original pero debemos seguir estrictamente los cambios planteados. Seguiremos esta pautas estrictamente a lo largo de todo el texto sin modificar nada más que lo que se indica a continuación:\n\n"
transicion = "\n\nAplicaremos estas pautas estrictamente sobre el texto que se proporcionará a continuación, manteniendo el tono y significado original del texto, pero realizando los ajustes necesarios para cumplir con todas las reglas establecidas sin alterarlo más allá. Si se incumplen las reglas, las partes que la incumplen deben de ser corregidas sin eliminarse totalmente (a menos que se requiera de su eliminación). Este es el texto original: "
adaptacion = "\n\nVisto el texto original, aquí a continuación tenemos el texto completo procesado: \n"


# Pruebas de distintas pautas con ejemplos
casos_prueba = [
    ["Las ideas enlazadas se deberían separar mediante un punto en lugar de una coma, cambiando el texto según sea necesario para mantener el sentido.",
     "Como ayer estaba lloviendo, Juan cogió el paraguas.",
     "Ayer estaba lloviendo. Por eso, Juan cogió el paraguas."],

    ["No se deberían usar las comillas (ni las españolas « » ni las inglesas “ ”, o presentadas como \" \"). Cuando se utilicen las comillas, como en las citas textuales, deben obligatoriamente ir acompañadas de una explicación del texto entrecomillado.",
     "El filósofo Aristóteles dijo “Lo que con mucho trabajo se adquiere, más se ama”.",
     "El filósofo Aristóteles dijo: “Lo que con mucho trabajo se adquiere, más se ama”. Esto quiere decir que valoramos más las cosas que cuestan esfuerzo."],

    ["Si se utilizan palabras con el mismo sonido o grafía (palabras homófonas u homógrafas) pero con distinto significado se debe redactar el texto de modo que facilite inequívocamente la comprensión del mismo.",
     "Juan coloca la maleta en la baca.",
     "Juan coloca la maleta en la baca del coche."],

    ["Si se utilizan palabras con el mismo sonido o grafía (palabras homófonas u homógrafas) pero con distinto significado se debe redactar el texto de modo que facilite inequívocamente la comprensión del mismo.",
     "Juan miraba los cascos que había en el suelo.",
     "Juan miraba los cascos de las botellas que había en el suelo."],

    ["Se debería evitar el uso de palabras muy largas o que contengan sílabas complejas.",
     "Pepa está la antepenúltima en la fila.",
     "Pepa está casi al final de la fila, tiene dos personas detrás."],

    ["Se deben evitar los adverbios terminados en –mente, sustituyéndolos por sinónimos o expresiones equivalentes que conserven el mismo significado. Asegúrate de no alterar el contexto ni la claridad de la oración.",
     "Él habla claramente.",
     "Él habla muy claro."],

    ["Se deben evitar los adverbios terminados en –mente, sustituyéndolos por sinónimos o expresiones equivalentes que conserven el mismo significado. Asegúrate de no alterar el contexto ni la claridad de la oración.",
     "Generalmente, Pedro come verduras.",
     "Pedro come verduras a menudo."],

    ["Cambiaremos todo superlativo en el texto por el adjetivo en su forma original junto con el adverbio \"muy\" delante. Por ejemplo, \"fortísimo\" sería cambiado por \"muy fuerte\".",
     "Grandísimo.",
     "Muy grande."],

    ["Es preferible evitar los verbos en forma nominal si crean expresiones abstractas. En su lugar, se deben usar verbos en conjugaciones claras para una mayor comprensión.",
     "María sube las escaleras hacia arriba.",
     "María sube las escaleras."],

    ["No se aceptan abreviaturas, si hay una palabra abreviada la cambiaremos a su forma original.",
     "Luis vive en la Avda. de Europa.",
     "Luis vive en la Avenida de Europa."],

    ["No se aceptan siglas que no sean acrónimos, estas se cambiarán por las palabras completas. En caso de que sean acrónimos de uso extendido, se deberá explicar su significado la primera vez que aparezcan.",
     "El próximo mes de agosto empezarán los JJOO.",
     "El próximo mes de agosto empezarán los Juegos Olímpicos."],

    ["Se debe usar siempre la misma palabra para referirse al mismo objeto o concepto a lo largo del texto, evitando sinónimos que puedan confundir.",
     "Juan compró dos pantalones de colores distintos. El rojo es más bonito.",
     "Juan compró dos pantalones de colores distintos. El pantalón rojo es más bonito."],

    ["No se debe usar sentido figurado como frases hechas o metáforas. Si es necesario, se puede añadir una explicación clara del significado.",
     "Ese libro cuesta un ojo de la cara.",
     "Ese libro cuesta mucho dinero."],

    ["No se debe usar sentido figurado como frases hechas o metáforas. Si es necesario, se puede añadir una explicación clara del significado.",
     "El frotar se va a acabar.",
     "Ya no es necesario frotar."],

    ["Es preferible evitar los verbos en forma nominal si crean expresiones abstractas. En su lugar, se deben usar verbos en conjugaciones claras para una mayor comprensión.",
     "El desayunar bien es importante.",
     "Es importante desayunar bien."],

    ["Es preferible evitar los verbos en forma nominal si crean expresiones abstractas. En su lugar, se deben usar verbos en conjugaciones claras para una mayor comprensión.",
     "El perder las llaves puso nervioso a Juan.",
     "Juan perdió las llaves y se puso nervioso."],

    ["Se debe usar siempre la misma palabra para referirse al mismo objeto o concepto a lo largo del texto, evitando sinónimos que puedan confundir.",
     "Lola tiene que rellenar una solicitud para hacer la obra. El formulario está en la página web del Ayuntamiento.",
     "Lola tiene que rellenar una solicitud para hacer la obra. La solicitud está en la página web del Ayuntamiento."],

    ["Se deben evitar los números ordinales (como 1.°) y usar cardinales (como 1) para una mayor claridad.",
     "Luis vive en la planta 18ª.",
     "Luis vive en la planta número 18."],

    ["Se deben evitar los números ordinales (como 1.°) y usar cardinales (como 1) para una mayor claridad.",
     "Juan vive en la planta decimoctava.",
     "Juan vive en la planta número 18."],

    ["Siempre que sea posible, se deben evitar oraciones impersonales, añadiendo un sujeto a la oración.",
     "Se enviará el formulario por correo electrónico.",
     "El ciudadano enviará el formulario por correo electrónico."],

    ["Es preferible evitar los tiempos verbales compuestos, condicionales y subjuntivos, usando en su lugar formas verbales simples.",
     "Jorge estaba cantando en su casa cuando sonó el timbre.",
     "Jorge cantaba en su casa y sonó el timbre."],

    ["Es preferible evitar los tiempos verbales compuestos, condicionales y subjuntivos, usando en su lugar formas verbales simples.",
     "La vecina quiere seguir viajando.",
     "La vecina quiere viajar más."],

    ["Es preferible usar oraciones afirmativas en lugar de negativas, salvo en casos de prohibiciones sencillas.",
     "No podrás marcharte antes del final de la reunión.",
     "Podrás marcharte cuando termine la reunión."],

    ["Se debe utilizar frases sencillas y evitar oraciones complejas. Si es necesario, se recomienda dividir las ideas en líneas distintas.",
     "Lo que dice Pedro, si he de ser sincero, me parece una tontería.",
     "Me parece que Pedro dice una tontería."],

    ["Se deben evitar explicaciones entre comas que interrumpan la fluidez de la oración. Si pueden omitirse sin afectar al significado, es mejor hacerlo.",
     "María, enfermera de profesión, puso la vacuna.",
     "María es enfermera y puso la vacuna."],

    ["Es preferible usar oraciones afirmativas en lugar de negativas, salvo en casos de prohibiciones sencillas.",
     "Luis sabía la respuesta. Sin embargo, se quedó callado.",
     "Luis sabía la respuesta, pero se quedó callado."],

    ["El imperativo debe utilizarse solo en contextos claros, evitando así confusión con la tercera persona del presente.",
     "Meter el pollo en el horno a 180°.",
     "Mete el pollo en el horno a 180 grados."],

    ["Para intentar usar un lenguaje inclusivo, se debe recurrir a términos genéricos, desdoblamientos claros o a la palabra 'persona' delante del adjetivo o nombre, evitando caracteres especiales como @ o desdoblamientos con barras.",
     "Los abogados/as presentaron una denuncia.",
     "Las abogadas y los abogados presentaron una denuncia."],

    ["Es preferible usar oraciones afirmativas en lugar de negativas, salvo en casos de prohibiciones sencillas.",
     "Los familiares de Pedro fueron a la reunión.",
     "La familia de Pedro fue a la reunión."],

    ["Para intentar usar un lenguaje inclusivo, se debe recurrir a términos genéricos, desdoblamientos claros o a la palabra 'persona' delante del adjetivo o nombre, evitando caracteres especiales como @ o desdoblamientos con barras.",
     "Los opositores estaban muy nerviosos.",
     "Las personas opositoras estaban muy nerviosas."]
]


# Generacion de texto para cada pauta con ejemplo
for caso in casos_prueba:
    prompt = caso[0]
    ejemplo_incorrecto = caso[1]
    ejemplo_correcto = caso[2]

    texto_final = prompt + transicion + ejemplo_incorrecto + adaptacion
    output = generator(texto_final, **generation_args)
    texto_generado = output[0]["generated_text"]

    print(f"Prompt: {prompt}")
    print(f"Ejemplo incorrecto: {ejemplo_incorrecto}")
    print(f"Ejemplo correcto: {ejemplo_correcto}")
    print(f"Texto generado: {texto_generado.removeprefix(texto_final).lstrip()}")
    print("\n---\n")
