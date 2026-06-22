####################################### 7 Classes #######################################
class_names_7 = [
'happiness.',
'sadness.',
'neutral.',
'anger.',
'surprise.',
'disgust.',
'fear.'
]
class_names_with_context_7 = [
'an expression of happiness.',
'an expression of sadness.',
'an expression of neutral.',
'an expression of anger.',
'an expression of surprise.',
'an expression of disgust.',
'an expression of fear.'
]
#AUs
class_descriptor_7 = [
'Cheek Raiser, Lip Corner Puller.',
'Inner Brow Raiser, Brow Lowerer, Lip Corner Depressor.',
'Relaxed Muscles, Even Eyebrows, Closed Lips, Calm Eyes, Smooth Forehead.',
'Brow Lowerer, Upper Lid Raiser, Lid Tightener, Lip Tightener.',
'Inner Brow Raiser, Outer Brow Raiser, Upper Lid Raiser, Jaw Drop.',
'Nose Wrinkler, Lip Corner Depressor, Lower Lip Depressor.',
'Inner Brow Raiser, Outer Brow Raiser, Brow Lowerer, Upper Lid Raiser, Lid Tightener, Lip Stretcher, Jaw Drop.',
]
# class_descriptor_7 = [
# 'a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.',
# 'tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.',
# 'relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.',
# 'furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.',
# 'widened eyes, an open mouth, raised eyebrows, and a frozen expression.',
# 'a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.',
# 'raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
# ]
# class_descriptor_7 = [
# "Cheek Raiser, Lip Corner Puller. Élévation des joues, traction des coins des lèvres.",
# "Inner Brow Raiser, Brow Lowerer, Lip Corner Depressor. Élévation des sourcils intérieurs, abaissement des sourcils, abaissement des coins des lèvres.",
# "Relaxed Muscles, Even Eyebrows, Closed Lips, Calm Eyes, Smooth Forehead. Muscles détendus, sourcils alignés, lèvres fermées, yeux calmes, front lisse.",
# "Brow Lowerer, Upper Lid Raiser, Lid Tightener, Lip Tightener. Abaissement des sourcils, élévation de la paupière supérieure, resserrement des paupières, resserrement des lèvres.",
# "Inner Brow Raiser, Outer Brow Raiser, Upper Lid Raiser, Jaw Drop. Élévation des sourcils intérieurs, élévation des sourcils extérieurs, élévation de la paupière supérieure, abaissement de la mâchoire.",
# "Nose Wrinkler, Lip Corner Depressor, Lower Lip Depressor. Plissement du nez, abaissement des coins des lèvres, abaissement de la lèvre inférieure.",
# "Brows Raiser, Upper Lid Raiser, Lid Tightener, Lip Stretcher, Jaw Drop. Élévation des sourcils, élévation de la paupière supérieure, resserrement des paupières, étirement des lèvres, abaissement de la mâchoire.",
# ]
# class_descriptor_7 = [
# 'A facial Expression characterized by raised cheeks and upward-curving lip corners, forming a smile with slight crinkling around the eyes. A relaxed upper face without tension in the brow or lips.',
# 'Raised inner brows, lowered outer brows, and downturned lip corners, forming a subtle or deep frown. The eyes appear slightly narrowed, reflecting sorrow with a subdued and drooping quality.',
# 'A facial Expression defined by relaxed facial muscles, even eyebrows, closed lips, calm eyes, and a smooth forehead. No visible tension.',
# 'A facial Expression marked by deeply furrowed brows, raised upper eyelids, and tightly pressed lips. The expression conveys frustration , with sharp, focused features and high tension, particularly in the brows and lips.',
# 'A facial Expression characterized by raised inner and outer brows, wide-open eyes with lifted upper lids, and a dropped jaw. It reflects shock or amazement.',
# 'A facial features a wrinkled nose, slightly raised upper lip, and downturned lip corners, sometimes with a dropped lower lip. The expression conveys aversion , distinguished by the upward wrinkle of the nose.',
# 'A facial Expression marked by raised brows, wide-open eyes with tightened lids, and lips that stretch horizontally or drop slightly. The expression reflects anxiety , with a strained and tense appearance.',
# ]

# class_descriptor_7 = [
# 'a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.',
# 'tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.',
# 'relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.',
# 'furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.',
# 'widened eyes, an open mouth, raised eyebrows, and a frozen expression.',
# 'a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.',
# 'raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
# ]

# class_descriptor_7 = [
# 'Upward movement of the corners of the mouth, relaxation of the jaw, narrowing of the eyes, raising of the eyebrows, crinkling of the eyes, brightening of the eyes.',
# 'Lowering of the corners of the mouth, drooping of the eyelids, narrowing of the eyes, furrowing of the eyebrows, tightening of the jaw.',
# 'Lips may be slightly parted, closed. Pupils may be slightly dilated, constricted. Eyebrows are typically horizontal, slightly raised.',
# 'Furrowing of eyebrows, narrowing of eyes, tightening of the jaw, lowering of eyebrows, clenching of teeth.',
# 'Widening of eyes, raising of eyebrows, parting of lips, slight opening of the mouth, raising of the cheeks.',
# 'Wrinkling of the nose, raising of the upper lip, lowering of the lower lip, squinting of the eyes, wrinkling of the cheeks, turning away of the head.',
# 'Widening of eyes, raising of eyebrows, parting of lips, tensing of the jaw, flaring of nostrils.',
# ]
# class_descriptor_7 = [
# 'The eyes gradually squint, the corners of the mouth lift. The eyebrows slightly raise, then relax, while the expression broadens, revealing teeth.',
# 'The eyebrows slant upward toward the center, with the inner corners slightly raised. The eyes appear slightly drooped, the mouth may turn downward at the corners.',
# 'The eyebrows in a neutral position, the eyes maintaining a calm and steady gaze, the mouth closed, slightly open.',
# 'The eyes narrow into a piercing glare, with the eyebrows drawing tightly together and downwards. The mouth tenses and may press into a firm line, open slightly.',
# 'The eyebrows are raised high, creating horizontal lines on the forehead. The eyes are wide open, the mouth is often slightly open.',
# 'The eyes may squint slightly, while the eyebrows lower, draw together. The nose wrinkles, the upper lip raises, exposing the teeth slightly as the mouth tightens.',
# 'The eyes widen with raised upper eyelids, and the eyebrows draw together, creating horizontal lines on the forehead. The mouth opens slightly, with the lips tensed.',
# ]










####11####
# class_descriptor_7 = [
# 'upturned mouth corners, lifted cheeks, and slightly squinted eyes.',
# 'downturned mouth corners, lowered eyebrows, and drooping eyelids.',
# 'relaxed mouth, level eyebrows, and open eyes.',
# 'furrowed eyebrows, narrowed eyes, flared nostrils, and a tense mouth.',
# 'raised eyebrows, wide-open eyes, and a slightly open mouth.',
# 'a wrinkled nose, raised upper lip, and slightly narrowed eyes.',
# 'wide-open eyes, raised eyebrows, and a slightly open mouth with tense lips.',
# ]
# class_descriptor_7 = [
# 'A smiling mouth, raised cheeks, sparkling eyes, and gently raised eyebrows.', 
# 'Moist eyes, a downturned mouth, slightly drooping eyelids, and furrowed eyebrows.',
# 'Relaxed features, a straight mouth, a smooth forehead, and neutral eyebrows.', 
# 'Furrowed brows, narrowed eyes, pressed lips, and flared nostrils.', 
# 'WWidened and rounded eyes, an open mouth with slightly parted lips, raised and high-set eyebrows, and an alert expression.',
# 'a scrunched nose, lowered eyebrows, a pursed mouth, and narrowed eyes.',
# 'Raised eyebrows, parted lips, tense eyelids, and a retracted chin.', 
# ]

####22####
#'A person with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows, showing happy.',
#'A person with downward pointing corners of the mouth, drooping eyelids, slightly furrowed brows, and a generally downwards and subdued look, showing sad.',
#'A person with relaxed eyebrows, eyes in a natural and unfocused state, a closed or slightly open mouth without tension, showing neutral.',
#'A person with lowered and furrowed eyebrows, narrowed and glaring eyes, flared nostrils, a mouth either firmly pressed or snarling, and a tensed jaw, showing anger.',
#'A person with raised eyebrows, wide-open eyes, a dropped jaw with the mouth open, showing surprise.',
#'A person with a wrinkled nose, raised upper lip, narrowed or squinting eyes, and a slightly open or curled lip, showing disgust.',
#'A person with wide-open eyes, raised and drawn together eyebrows, a tensed or slightly open mouth, and a generally stretched or elongated face, showing fear.',
# class_descriptor_7 = ['A smiling mouth, raised cheeks, sparkling eyes, and gently raised eyebrows.', #A smiling mouth with upturned corners, raised and rounded cheeks, crinkled eyes with crow's feet, and gently arched eyebrows.

#####33###
# class_descriptor_7 = [
# 'A smiling mouth with upturned corners, raised and rounded cheeks, crinkled eyes with crows feet, and gently arched eyebrows.',
# 'Moist eyes with a slight sheen, a downturned mouth with a quivering lower lip, slightly drooping upper eyelids, and furrowed eyebrows.',
# 'Relaxed features, a straight and closed mouth, a smooth and unwrinkled forehead, and eyebrows in a natural.',
# 'Furrowed and drawn-together eyebrows, narrowed and piercing eyes, tightly pressed lips, and flared nostrils.',
# 'Widened and rounded eyes, an open mouth with slightly parted lips, raised and high-set eyebrows, and an alert expression.',
# 'A wrinkled nose, lowered eyebrows, a pursed  mouth, and narrowed eyes.',
# 'Raised and arched eyebrows, parted lips with slightly open mouth, tense and wide-open eyelids, and a retracted chin pulled slightly back.',
# ]

####################################### 11 Classes #######################################
class_names_11 = [
'happiness.',
'sadness.',
'neutral.',
'anger.',
'surprise.',
'disgust.',
'fear.',
'contempt.',
'anxiety.',
'helplessness.',
'disappointment.'
]
class_names_with_context_11 = [
'an expression of happiness.',
'an expression of sadness.',
'an expression of neutral.',
'an expression of anger.',
'an expression of surprise.',
'an expression of disgust.',
'an expression of fear.',
'an expression of contempt.',
'an expression of anxiety.',
'an expression of helplessness.',
'an expression of disappointment.',
]
class_descriptor_11 = [
'a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.',
'tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.',
'relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.',
'furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.',
'widened eyes, an open mouth, raised eyebrows, and a frozen expression.',
'a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.',
'raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
'one side of its mouth raised, one eyebrow lower and one raised, narrowed eyes, and a raised chin.',
'a tensed forehead, tightly pressed lips, pupil dilation, and tensed facial muscles.',
'drooping eyebrows, a downward gaze, a downturned mouth, and lacking expression.',
'a downturned mouth, lowered eyebrows, narrowed eyes, and a sighing face.'
]