"""
GPT-4 prompt templates for open-set scene generation.
Prompts are copied verbatim from LayoutVLM paper Appendix B.1.
"""

# Prompt 1: Layout Criteria Generation
LAYOUT_CRITERIA_PROMPT = """Given a task description, return a string description of layout criteria for an interior design focused on the provided task.
Include considerations for aesthetics, functionality, and spatial organization. Each layout criteria string should start with
the phrase "The layout criteria should follow the task description and be...".
For example, if the task description is a spatious study room, the layout criteria should be:
"The layout criteria should follow the task description and be spatious, tidy, and minimal"
task description: {task_description}
Return only the layout criteria and nothing else. Ensure that the criteria is
no longer than 1-2 sentences. It is extremely important."""

# Prompt 2: Asset List Generation
ASSET_LIST_PROMPT = """Given a client's description of a room or tabletop and the floor vertices of the room, determine what objects and how many of
them should be placed in this room or table.
Objects should follow the requirements below.
Requirement 1: Be specifc about how you describe the objects, while using as simple english as possible. Each object should try to be around two words, including
a relevant adjective if possible. Normally this adjective should be the room type, such as "kitchen scale" or "bathroom sink". However, it can also be a color or
a material, such as "wooden chair" or "red table" if necessary. Ensure that descriptions are simple - an elementary student should understand what each object is.
For example, if a client description asks for "weightlifting racks", simplify the description to "weightlifting equipment".
For example, if a client description asks for "a 1980's jukebox", simplify the description to "vintage jukebox".
For example, if a client description includes "aerobic machines",simplify the description to "treadmill" and "exercise bike".
For example, if a client is describing a kitchen and is asking for a "scale" for food, ensure the object includes an adjective to describe the object, such as "kitchen scale".
For example, if a client is describing a bathroom and is asking for a "sink", ensure the object includes an adjective to describe the object, such as "bathroom sink".
Requirement 2: Only choose objects that are singular in nature.
For example, instead of choosing a "speaker system", just choose "speaker".
For example, Instead of choosing "tables" and "chairs", just choose "table" and "chair".
Requirement 3: Ensure that the objects are relevant to the room or tabletop.
A client's description can either describe a room or tabletop arrangement. If it is describing a tabletop arrangement,
do not include objects like "table" or "chair" in the response. Only include objects that would be placed on the table.
If it is describing a room arrangement, do not describe include things like "windows" or "doors" in the response.
Only include objects that would be placed in the room. Other than paintings, posters, light fixtures, or shelfs,
do not include objects that would be placed on the wall.
Requirement 4: Ensure that rooms have a place to sit and a place to put things down, like a counter, table, or nightstand.
This also means that objects like art easels, work benches, or desks should have corresponding chairs or stools.
For example, if a client is describing a bar, ensure that the response includes a "bar table" or "counter" and "bar stools" as well.
For example, if a client describes a classroom, ensure that all desks have corresponding chairs.
Requirement 5: Try and include as many objects as possible that are relevant to the room or tabletop. Aim for at least 10
objects in each response, but ideally include more.
After ensuring these requirements, return a dictionary objects, where the key is the object name and the value is an array tuple of two values.
The first value of the key array is the number of times that object should occur in the room and the second value is how many
types of that object should exist.
For example, for a given description of a garden, you would want many plants, but do not want all of them to be the same type.
Thus, the value of the key array would be [9, 3] for the object "plant". This means that there should be 9 plants in the garden
and there should be 3 different types of ferns in the garden.
For example, for a given description of "A study room 5m x 5m"
Return the Dictionary: {{"desk": [1, 1], "chair": [1, 1], "lamp": [1, 1], "bookcase": [2, 1], "laptop_computer": [1, 1], "computer monitor": [1, 1], "printer": [1, 1], "sofa": [1, 1], "flowerpot": [1, 1], "painting": [1, 1]}}
For example, for a given description of "A tabletop arrangement with a bowl placed on a plate 1m x 1m"
Return the Dictionary: {{"plate": [1, 1], "bowl": [1, 1], "fork": [1, 1], "knife": [1, 1], "spoon": [1, 1], "napkin": [1, 1], "salt shaker": [1, 1], "pepper shaker": [1, 1], "wine glass": [1, 1], "water glass": [1, 1]}}
For example, for a given description of "a vibrant game room filled with vintage arcade games and a jukebox, 6m x 6m"
Return the Dictionary: {{"jukebox": [1, 1], "arcade machine": [3, 1], "pool table": [1, 1], "darts board": [1, 1], "bar stool": [4, 1], "bar table": [1, 1], "neon sign": [1, 1], "popcorn machine": [1, 1], "vending machine": [1, 1], "air hockey table": [1, 1]}}
For example, for a given description of "a lush inside garden filled with a variety of plants and a small birdbath, 5m x 3m"
Return the Dictionary: {{"fern": [8, 3], "birdbath": [1, 1], "flowerpot": [3, 1], "watering can": [1, 1], "garden gnome": [1, 1], "garden bench": [1, 1], "garden shovel": [1, 1], "garden rake": [1, 1], "garden hose": [1, 1]}}
task description: {task_description}
layout criteria: {layout_criteria}
room size in meters: {room_size}
Remember, you should only include objects that are most important to be placed in the room or on the table.
The dictionary should not include the room dimensions.
Return only the dictionary of objects and nothing else. It is extremely important."""

# Prompt 3: Asset Verification
ASSET_VERIFICATION_PROMPT = """You are an interior designer. A client is suggesting possible objects that he thinks belongs in a described
room. You are tasked with determining if the client is correct or not, stating whether the proposed object
belongs in the described room.
Given a client's description of a room or tabletop, the description of an object, and images of the object,
determine if the described object should be placed in the room that is described. To help, you are also
given a description of what object the client was initially looking for. Ensure that the style and color
of the object matches the type of the room. If an object is not in the style of what the room type
would typically have, it should not be placed in the room.
Return "True" if the object should be kept in the room and "False" if the object should not be.
For example, if the room description is a "A tabletop arrangement with a bowl placed on a plate 1m x 1m" and the object appears to be "a shovel":
Return: False
For example, if the room description is a "A spatious study room with a desk and chair" and the object appears to be "an 18th century book":
Return: True
For example, for a given description of "a vibrant game room filled with vintage arcade games and a jukebox, 6m x 6m" and the object appears to be "a 1980s pinball machine":
Return: True
For example, for a given description of an "art room with chairs", and the object appears to be a "a pink beach chair":
Return: False
task description: {task_description}
layout criteria: {layout_criteria}
object description: {object_description}
object client requested: {object_looking_for}
Remember, you should only return "True" if the object should be placed in the room / tabletop and "False" if the object should not be.
Do not include any other words in your response. It is extremely important."""

# Prompt 4: Room Dimension Estimation (inspired by Holodeck)
ROOM_DIMENSIONS_PROMPT = """You are an experienced interior designer. Given a room description, estimate appropriate room dimensions in meters.

Guidelines:
1. Room dimensions (width and depth) should typically be between 3m and 8m each.
2. Maximum room area should not exceed 48 square meters.
3. Consider the room type and contents when estimating size:
   - Bedrooms: typically 3-5m per side
   - Living rooms: typically 4-6m per side
   - Kitchens: typically 3-4m per side
   - Dining rooms: typically 3-5m per side
   - Bathrooms: typically 2-3m per side
   - Offices/studies: typically 3-4m per side
4. If specific dimensions are mentioned in the description (e.g., "5m x 5m"), use those exactly.
5. Consider the number and size of objects mentioned - more furniture requires more space.

Room description: {task_description}

Return ONLY a JSON object with width and depth in meters, like this:
{{"width": 4.0, "depth": 5.0}}

Do not include any other text."""
