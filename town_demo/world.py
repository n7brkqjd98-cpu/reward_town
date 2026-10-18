"""Static world data for Reward Town: map, personas, actions, knowledge, conversation topics."""

from dataclasses import dataclass

MAP_W = 40
MAP_H = 40


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    kind: str  # "home" | "public"
    door: tuple[int, int] | None  # buildings: the door tile on the Tuxemon map
    area: tuple[int, int, int, int] | None  # open areas: (x, y, w, h)


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    initials: str
    age: int
    color: str
    home: str
    wake: str  # "HH:MM"
    description: str
    lifestyle: str
    sprite: str  # Tuxemon overworld sprite / portrait name


@dataclass(frozen=True)
class Action:
    id: str
    text: str
    emoji: str
    location: str  # location id or "home"
    hours: tuple[tuple[int, int], ...]  # [start, end) hour ranges
    duration: int  # ticks
    max_per_day: int
    agents: tuple[str, ...] | None = None
    requires_flag: str | None = None
    tool: str | None = None


@dataclass(frozen=True)
class Line:
    text: str
    effect: str | None = None


@dataclass(frozen=True)
class Topic:
    id: str
    candidate_text: str
    initiators: tuple[str, ...] | None
    requires_flag: str | None
    requires_knowledge: str | None
    responder_lacks: str | None
    turns: tuple[tuple[Line, ...], ...]
    effects: dict[str, tuple[frozenset[str], frozenset[str]]]
    excluded_initiators: tuple[str, ...] = ()


@dataclass(frozen=True)
class Technique:
    id: str  # "<tool id>_<index>", e.g. "espresso_machine_3"
    text: str  # verb phrase following the user's name


@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    emoji: str
    location: str  # location id
    tile: tuple[int, int] | None  # outdoor tools only: the map tile the player faces and presses A on
    task: str
    product: str
    techniques: tuple[Technique, ...]  # exactly 6; [0] and [1] are the basics everyone knows


@dataclass(frozen=True)
class Stage:
    id: str  # e.g. "arch_frame"
    name: str  # short noun used in UI/prompts, e.g. "frame"
    tool: str  # tool id
    task: str  # infinitive phrase after "to", used in the work/reaction prompts
    product: str  # what the judge rates, e.g. "arch frame"
    text: str  # verb phrase after the builder's name, used in decisions/events
    duration: int  # ticks


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    location: str  # location id
    tiles: tuple[tuple[int, int], ...]  # tiles the arch stands on; the player faces/stands on one and presses A
    hours: tuple[int, int]  # [start, end) hours when building may happen
    stages: tuple[Stage, ...]


LOCATIONS = (
    Location("lin_house", "Lin family house", "home", (27, 26), None),
    Location("moreno_house", "Moreno house", "home", (20, 34), None),
    Location("moore_house", "Moore house", "home", (25, 8), None),
    Location("isabella_apt", "Isabella's apartment", "home", (35, 26), None),
    Location("hobbs_cafe", "Hobbs Cafe", "public", (31, 16), None),
    Location("willow_market", "The Willows Market and Pharmacy", "public", (30, 34), None),
    Location("pub", "The Rose and Crown Pub", "public", (16, 16), None),
    Location("college", "Oak Hill College", "public", (17, 9), None),
    Location("johnson_park", "Johnson Park", "public", None, (5, 21, 9, 8)),
    Location("dorm", "Oak Hill College dorm", "home", (20, 26), None),
)
LOC_BY_ID = {loc.id: loc for loc in LOCATIONS}

PERSONAS = (
    Persona(
        "isabella", "Isabella Rodriguez", "IR", 34, "#e6194b", "isabella_apt", "06:00",
        "the friendly owner of Hobbs Cafe who loves making people feel welcome. She is planning a "
        "Valentine's Day party at Hobbs Cafe today from 5pm to 7pm and wants to invite as many people as possible.",
        "Isabella goes to bed around 11pm, wakes up around 6am, and spends most of the day running the cafe.",
        "barmaid",
    ),
    Persona(
        "klaus", "Klaus Mueller", "KM", 20, "#3cb44b", "dorm", "07:00",
        "a sociology student at Oak Hill College who is passionate about social justice. He is writing a "
        "research paper on the effects of gentrification in low-income communities and his deadline is very close.",
        "Klaus goes to bed around 11pm, wakes up around 7am, and spends most of the day studying.",
        "cooldude",
    ),
    Persona(
        "maria", "Maria Lopez", "MaL", 21, "#4363d8", "dorm", "07:30",
        "a physics student at Oak Hill College who streams video games on Twitch in her spare time. She is "
        "outgoing, loves meeting new people, and has a secret crush on Klaus Mueller.",
        "Maria goes to bed around midnight, wakes up around 7:30am, attends classes during the day and streams "
        "in the evening.",
        "goth",
    ),
    Persona(
        "john", "John Lin", "JL", 45, "#f58231", "lin_house", "06:00",
        "a pharmacist at The Willows Market and Pharmacy who loves to help people. He lives with his wife Mei "
        "and his son Eddy.",
        "John goes to bed around 10pm, wakes up around 6am, and works at the pharmacy from 8am to 5pm.",
        "scientist",
    ),
    Persona(
        "mei", "Mei Lin", "MeL", 44, "#911eb4", "lin_house", "06:30",
        "a philosophy professor at Oak Hill College who loves discussing ethics. She is married to John Lin and "
        "is proud of her son Eddy.",
        "Mei goes to bed around 10:30pm, wakes up around 6:30am, teaches during the day and grades essays in "
        "the evening.",
        "florist",
    ),
    Persona(
        "eddy", "Eddy Lin", "EL", 19, "#42d4f4", "lin_house", "07:30",
        "a music composition student at Oak Hill College who is working on a piano piece for his class. He is "
        "creative, a little shy, and loves exploring new musical styles.",
        "Eddy goes to bed around midnight, wakes up around 7:30am, and spends his free time composing.",
        "postboy",
    ),
    Persona(
        "tom", "Tom Moreno", "TM", 50, "#f032e6", "moreno_house", "05:30",
        "the owner of The Willows Market and Pharmacy. He is hard-working, a bit grumpy before his coffee, and "
        "knows most of the town by name.",
        "Tom goes to bed around 10pm, wakes up around 5:30am, and runs the market from 7am to 7pm.",
        "shopassistant",
    ),
    Persona(
        "sam", "Sam Moore", "SM", 65, "#9a6324", "moore_house", "06:30",
        "a retired navy officer who is running for mayor in the local election. He wants to meet as many "
        "residents as possible and tell them about his plans for the town.",
        "Sam goes to bed around 10pm, wakes up around 6:30am, and spends the day campaigning around town.",
        "riverboatcaptain",
    ),
)
PERSONA_BY_ID = {p.id: p for p in PERSONAS}

PARK_SPOTS = {"isabella": (9, 22), "klaus": (7, 22), "maria": (11, 22), "john": (6, 25), "mei": (12, 25),
              "eddy": (9, 26), "tom": (6, 27), "sam": (12, 27)}
PLAYER_START = (25, 38)  # just inside the Route 1 entrance, with an open path north
PLAYER_SPRITES = {"boy": "adventurer", "girl": "heroine"}


def _a(id, text, emoji, location, hours, duration, max_per_day, agents=None, requires_flag=None, tool=None):
    return Action(id, text, emoji, location, tuple(tuple(h) for h in hours), duration, max_per_day,
                  agents, requires_flag, tool)


def _tool(id, name, emoji, location, tile, task, product, *texts) -> Tool:
    return Tool(id, name, emoji, location, tile, task, product,
                tuple(Technique(f"{id}_{i}", text) for i, text in enumerate(texts)))


TOOLS = (
    _tool("espresso_machine", "espresso machine", "☕", "hobbs_cafe", None, "make a latte for a customer", "latte",
          "presses the default latte button and hands over the cup",
          "pulls a quick shot and pours in milk straight from the fridge",
          "grinds fresh beans, tamps them evenly and pulls a 25-second shot",
          "steams the milk until it is silky and just under 65°C",
          "pours the milk slowly to draw a heart in the foam",
          "dials in the grind for today's humidity, pulls a balanced double shot and finishes with a tulip of "
          "microfoam"),
    _tool("oven", "oven", "🍪", "hobbs_cafe", None, "bake heart-shaped cookies for the Valentine's Day party",
          "batch of heart-shaped cookies",
          "throws the dough in and sets the oven to the highest temperature to save time",
          "bakes the cookies straight from a store-bought tub of dough",
          "preheats the oven to 175°C and lines the trays with parchment",
          "chills the dough first so the hearts keep their shape",
          "rotates the trays halfway through so every cookie bakes evenly",
          "pulls the cookies while the centers are still soft, lets them finish on the hot tray and pipes pink "
          "icing hearts"),
    _tool("pill_counter", "pill counter", "💊", "willow_market", None, "fill a prescription", "prescription",
          "counts the pills by hand from memory",
          "pours pills onto the tray and eyeballs the amount",
          "checks the prescription against the patient's records before counting",
          "counts the pills in fives on the tray and double-checks the total",
          "verifies the dosage and checks for drug interactions",
          "counts twice, prints a clear label, and explains the dosage and side effects to the patient"),
    _tool("shelf_cart", "shelf cart", "📦", "willow_market", None, "restock the shelves", "shelf display",
          "dumps the new boxes in front of the old stock",
          "stacks everything on the nearest empty shelf",
          "moves older items to the front so they sell first",
          "lines up the labels so every product faces the customer",
          "checks expiry dates and pulls anything out of date",
          "rotates stock first-in-first-out, faces every label and builds an eye-catching Valentine's display by "
          "the entrance"),
    _tool("piano", "piano", "🎹", "lin_house", None, "compose a new piece for music class", "piano piece",
          "plays the same four chords over and over",
          "copies the melody of a pop song from the radio",
          "improvises a melody and records it on a phone",
          "writes a chord progression first and builds a melody over it",
          "develops a short motif and varies it in each section",
          "sketches a theme, modulates it to a new key in the bridge and writes out the score with dynamics"),
    _tool("library_computer", "library computer", "💻", "college", None,
          "write a section of a research paper on gentrification", "research section",
          "copies a paragraph from a website and moves on",
          "writes from memory without checking any sources",
          "searches the library database for peer-reviewed studies",
          "takes notes from three sources and compares their findings",
          "builds an outline and cites every claim",
          "combines census data with interview quotes, cites every source and revises the argument for clarity"),
    _tool("lab_bench", "lab bench", "🔬", "college", None, "run a physics experiment", "experiment",
          "takes a single measurement and calls it done",
          "skips calibration because it takes too long",
          "calibrates the instruments before measuring",
          "repeats the measurement five times and averages the results",
          "estimates the uncertainty of every measurement",
          "calibrates, takes repeated measurements, plots them with error bars and checks the result against "
          "theory"),
    _tool("megaphone", "megaphone", "📣", "johnson_park", (9, 23), "give a campaign speech in the park",
          "campaign speech",
          "shouts a slogan into the megaphone at full volume",
          "reads a long list of statistics from a notepad",
          "introduces himself and explains why he is running",
          "tells a personal story about the town",
          "asks the crowd what they care about and answers them",
          "keeps the volume friendly, shares a short story, lays out three clear promises and invites questions"),
    _tool("watering_can", "watering can", "🌷", "johnson_park", (11, 28), "tend the flower beds", "flower bed",
          "dumps the whole can on one plant",
          "waters the leaves at noon in the full sun",
          "waters the soil at the base of each plant",
          "pulls the weeds before watering",
          "checks the soil moisture with a finger first",
          "waters deeply at the roots, mulches the bed and deadheads the faded blooms"),
)
# Tools for the wordless heart-arch build: no tile and no action; NPCs pick them up from the build option.
BUILD_TOOLS = (
    _tool("saw", "saw", "🪚", "johnson_park", None, "cut the posts and the beam for the Valentine's heart arch",
          "arch frame",
          "saws freehand without measuring anything",
          "uses the first board in the pile, knots and all",
          "measures twice and marks every cut with a pencil",
          "clamps the board down before cutting",
          "cuts slowly along the line with long, even strokes",
          "measures twice, clamps each board, cuts square along the marks and sands the edges smooth"),
    _tool("hammer", "hammer", "🔨", "johnson_park", None, "nail the heart arch together over the park's south gate",
          "arch structure",
          "hammers the nails in at whatever angle they land",
          "uses glue instead of nails to save time",
          "pre-drills holes so the wood doesn't split",
          "checks that the posts are level before nailing",
          "braces the corners with diagonal supports",
          "levels the posts, pre-drills, nails the beam with braced corners and tests that the arch is sturdy"),
    _tool("paintbrush", "paintbrush", "🖌️", "johnson_park", None, "paint the heart arch white with red hearts",
          "painted arch",
          "slaps on one thick coat of paint",
          "paints straight over the dust and splinters",
          "sands the wood before painting",
          "applies a primer first",
          "uses a stencil for the red hearts",
          "sands, primes, paints two thin white coats and stencils neat red hearts once they dry"),
    _tool("string_lights", "string lights", "💡", "johnson_park", None, "hang string lights along the heart arch",
          "lit-up arch",
          "drapes the lights in a tangled heap",
          "plugs in the lights without testing them",
          "tests every strand before hanging it",
          "spaces the clips evenly along the beam",
          "runs the cable safely down the back of the post",
          "tests each strand, clips them evenly in a heart-shaped curve and hides the cable behind the post"),
)
ALL_TOOLS = TOOLS + BUILD_TOOLS
TOOL_BY_ID = {t.id: t for t in ALL_TOOLS}
PROJECT = Project("heart_arch", "Valentine's heart arch", "johnson_park", ((8, 28), (9, 28), (10, 28)), (8, 18), (
    Stage("arch_frame", "frame", "saw", "cut the posts and the beam for the Valentine's heart arch", "arch frame",
          "saws the posts and beam for the Valentine's heart arch", 3),
    Stage("arch_assemble", "assembly", "hammer", "nail the heart arch together over the park's south gate",
          "arch structure", "hammers the heart arch together over the park's south gate", 3),
    Stage("arch_paint", "paint", "paintbrush", "paint the heart arch white with red hearts", "painted arch",
          "paints the heart arch white with red hearts", 3),
    Stage("arch_lights", "lights", "string_lights", "hang string lights along the heart arch", "lit-up arch",
          "hangs string lights along the heart arch", 2),
    Stage("arch_flowers", "roses", "watering_can", "plant roses at the foot of the heart arch and water them",
          "rose bed", "plants and waters roses at the foot of the heart arch", 2),
))
STAGE_BY_ID = {s.id: s for s in PROJECT.stages}
BUILD_MAX_PER_DAY = 2
PROJECT_STARTERS = ("tom", "sam")  # the NPCs who know about the arch from the start
BASIC_TECHNIQUES = 2  # indices 0 and 1 of every tool: known by everyone, the player included
SOLO_SKILL = 3  # a non-specialist may use a tool alone once it knows this many techniques for it
LEARNED_TOOL_MAX_PER_DAY = 2
INITIAL_TECHNIQUES: dict[str, dict[str, tuple[int, ...]]] = {  # indices on top of the basics
    "isabella": {"espresso_machine": (2, 3, 4), "oven": (2, 3), "megaphone": (5,), "string_lights": (5,)},
    "klaus": {"library_computer": (2, 3, 4), "lab_bench": (5,), "saw": (5,)},
    "maria": {"lab_bench": (2, 3, 4), "string_lights": (2, 3, 4)},
    "john": {"pill_counter": (2, 3, 4), "shelf_cart": (5,), "hammer": (5,)},
    "mei": {"watering_can": (2, 3, 4), "oven": (4, 5), "piano": (5,), "library_computer": (5,), "paintbrush": (5,)},
    "eddy": {"piano": (2, 3, 4), "paintbrush": (2, 3, 4)},
    "tom": {"shelf_cart": (2, 3, 4), "espresso_machine": (5,), "pill_counter": (5,), "saw": (2, 3, 4)},
    "sam": {"megaphone": (2, 3, 4), "watering_can": (5,), "hammer": (2, 3, 4)},
}
REACTIONS = (  # index k gives 4 - k stars
    "{o}: Wow, that's the best {product} I've seen in a while!",
    "{o}: Nicely done, that looks really good.",
    "{o}: Not bad, but it could be better.",
    "{o}: Hmm, I think something went wrong there.",
)
LEARN_LINES = (  # index 0: the technique is learned
    "{o} memorizes how {actor} {technique} to try it later.",
    "{o} doesn't pay much attention to how {actor} uses the {tool}.",
)
HELP_ANSWERS = ("{h}: Sure, I'd be happy to help!", "{h}: Sorry, I can't help right now.")  # index 0: accept
PLAYER_HELP_ANSWERS = ("Sure, I'll help!", "Sorry, not right now.")
# Median raw judge score of the technique NPCs chose in a headless day: quality 50 at this score.
TOOL_SCORE_OFFSET = -2.2


ACTIONS = (
    # Generic actions
    _a("sleep", "is sleeping", "😴", "home", [[0, 7], [21, 24]], 6, 99),
    _a("wake_up", "wakes up and gets out of bed", "🛏️", "home", [[5, 10]], 1, 1),
    _a("morning_routine", "takes a shower and gets dressed", "🚿", "home", [[5, 11]], 2, 1),
    _a("breakfast_home", "eats breakfast", "🥣", "home", [[6, 10]], 2, 1),
    _a("breakfast_cafe", "has breakfast and a coffee", "☕", "hobbs_cafe", [[7, 11]], 3, 1),
    _a("lunch_home", "cooks and eats lunch", "🍲", "home", [[11, 14]], 3, 1),
    _a("lunch_cafe", "eats lunch", "🥪", "hobbs_cafe", [[11, 14]], 4, 1),
    _a("groceries", "buys groceries", "🛒", "willow_market", [[9, 20]], 3, 1),
    _a("walk_park", "takes a walk", "🌳", "johnson_park", [[7, 19]], 3, 2),
    _a("dinner_home", "cooks and eats dinner", "🍝", "home", [[17, 21]], 4, 1),
    _a("pub_evening", "has a drink with friends", "🍺", "pub", [[18, 23]], 6, 1),
    _a("relax_home", "relaxes and reads a book", "📖", "home", [[0, 24]], 3, 99),
    _a("get_ready_bed", "gets ready for bed", "🪥", "home", [[20, 24]], 1, 1),
    _a("attend_party", "attends the Valentine's Day party", "🎉", "hobbs_cafe", [[17, 19]], 6, 1,
       requires_flag="party_accepted"),
    # Isabella
    _a("open_cafe", "opens the cafe and brews the first pots of coffee", "☕", "hobbs_cafe", [[6, 9]], 2, 1,
       ("isabella",)),
    _a("use_espresso", "makes lattes with the espresso machine", "☕", "hobbs_cafe", [[7, 20]], 3, 99,
       ("isabella",), tool="espresso_machine"),
    _a("use_oven", "bakes heart-shaped cookies in the oven for the party", "🍪", "hobbs_cafe", [[10, 17]], 3, 2,
       ("isabella",), tool="oven"),
    _a("decorate_cafe", "decorates the cafe with hearts and flowers for the Valentine's Day party", "💐",
       "hobbs_cafe", [[12, 17]], 3, 2, ("isabella",)),
    _a("host_party", "hosts the Valentine's Day party", "🎉", "hobbs_cafe", [[17, 19]], 6, 2, ("isabella",),
       requires_flag="host"),
    _a("close_cafe", "cleans up and closes the cafe", "🧹", "hobbs_cafe", [[19, 22]], 2, 1, ("isabella",)),
    # Klaus
    _a("use_library_computer", "researches gentrification on the library computer", "💻", "college",
       [[8, 20]], 6, 99, ("klaus",), tool="library_computer"),
    _a("write_paper_cafe", "writes his research paper on his laptop", "💻", "hobbs_cafe", [[9, 19]], 4, 2,
       ("klaus",)),
    _a("sociology_seminar", "attends a sociology seminar", "🎓", "college", [[9, 16]], 6, 1, ("klaus",)),
    # Maria
    _a("physics_lecture", "attends a physics lecture", "⚛️", "college", [[9, 15]], 6, 1, ("maria",)),
    _a("use_lab_bench", "runs a physics experiment at the lab bench", "🔬", "college", [[8, 21]], 4, 3,
       ("maria",), tool="lab_bench"),
    _a("stream_games", "streams video games on Twitch", "🎮", "home", [[14, 24]], 6, 2, ("maria",)),
    # John
    _a("use_pill_counter", "fills prescriptions with the pill counter", "💊", "willow_market", [[8, 18]], 6, 99,
       ("john",), tool="pill_counter"),
    # Mei
    _a("teach_class", "teaches a philosophy class", "🏫", "college", [[9, 16]], 6, 1, ("mei",)),
    _a("office_hours", "holds office hours for her students", "🗣️", "college", [[13, 17]], 3, 1, ("mei",)),
    _a("grade_essays", "grades student essays", "📝", "home", [[8, 23]], 4, 2, ("mei",)),
    _a("use_watering_can", "tends the flower beds with the watering can", "🌷", "johnson_park", [[7, 19]], 3, 2,
       ("mei",), tool="watering_can"),
    # Eddy
    _a("music_class", "attends music theory class", "🎼", "college", [[9, 15]], 6, 1, ("eddy",)),
    _a("use_piano", "composes on the piano", "🎹", "lin_house", [[8, 24]], 4, 2, ("eddy",), tool="piano"),
    _a("violin_park", "practices violin", "🎻", "johnson_park", [[10, 18]], 3, 1, ("eddy",)),
    # Tom
    _a("use_shelf_cart", "restocks the shelves with the shelf cart", "📦", "willow_market", [[7, 20]], 6, 99,
       ("tom",), tool="shelf_cart"),
    # Sam
    _a("use_megaphone", "gives a campaign speech with the megaphone", "📣", "johnson_park", [[9, 18]], 4, 2,
       ("sam",), tool="megaphone"),
    _a("campaign_cafe", "talks to voters about his mayoral campaign", "🗳️", "hobbs_cafe", [[8, 18]], 3, 2,
       ("sam",)),
    _a("write_speech", "writes his campaign speech", "✍️", "home", [[8, 22]], 4, 1, ("sam",)),
)
ACTION_BY_ID = {a.id: a for a in ACTIONS}

KNOWLEDGE = {
    "party": "Isabella Rodriguez is hosting a Valentine's Day party at Hobbs Cafe today from 5pm to 7pm.",
    "election": "Sam Moore is running for mayor in the local election.",
}
INITIAL_KNOWLEDGE = {"isabella": {"party"}, "sam": {"election"}}
INITIAL_FLAGS = {"isabella": {"host"}}
FLAG_TEXT = {"party_accepted": "{name} promised to come to the Valentine's Day party."}


def _lines(*texts: str) -> tuple[Line, ...]:
    return tuple(Line(t) for t in texts)


_PARTY_T1 = _lines(
    "Oh, a party? That sounds lovely. Who else is coming?",
    "Tonight? I'm not sure yet, I have a lot on my plate today.",
    "I hadn't heard about it! What time does it start again?",
    "That's sweet of you to mention it. Tell me more.",
)
_PARTY_T3 = (
    Line("Count me in! I'll be at Hobbs Cafe at 5.", "accept"),
    Line("I'd love to come. Thanks for telling me, {a}!", "accept"),
    Line("Sorry, I really need to work tonight, so I'll have to pass.", "decline"),
    Line("I don't think I can make it this time, but I hope it goes well.", "decline"),
)
_PARTY_EFFECTS = {
    "accept": (frozenset({"party"}), frozenset({"party_accepted"})),
    "decline": (frozenset({"party"}), frozenset()),
}

TOPICS = (
    Topic(
        id="party_invite",
        candidate_text="walks over to {b} and invites them to her Valentine's Day party",
        initiators=("isabella",),
        requires_flag="host",
        requires_knowledge=None,
        responder_lacks="party",
        turns=(
            _lines(
                "Hi {b}! I'm throwing a Valentine's Day party at Hobbs Cafe today from 5 to 7pm. "
                "Would you like to come?",
                "Hey {b}, how's your day going? I'm organizing a little Valentine's celebration at the cafe tonight.",
                "{b}, do you have plans this evening? There's a party at Hobbs Cafe at 5pm and everyone's welcome.",
                "Good to see you, {b}. Did you hear about the Valentine's Day party at Hobbs Cafe tonight?",
            ),
            _PARTY_T1,
            _lines(
                "It starts at 5pm. There will be music, snacks and lots of friendly faces from town.",
                "It's from 5 to 7pm at Hobbs Cafe. No need to bring anything, just come and enjoy yourself.",
                "Totally understand if you're busy, but even stopping by for a few minutes would be great.",
                "I'm inviting everyone I see today, so you'll know plenty of people there.",
            ),
            _PARTY_T3,
        ),
        effects=_PARTY_EFFECTS,
    ),
    Topic(
        id="party_word_of_mouth",
        candidate_text="tells {b} about Isabella's Valentine's Day party at Hobbs Cafe",
        initiators=None,
        requires_flag="party_accepted",
        requires_knowledge=None,
        responder_lacks="party",
        turns=(
            _lines(
                "Hey {b}, did you hear Isabella is hosting a Valentine's Day party at Hobbs Cafe tonight?",
                "{b}! Isabella invited me to a party at Hobbs Cafe at 5pm. You should come too.",
                "Are you going to Isabella's Valentine's Day party later, {b}?",
                "{b}, a bunch of us are heading to Hobbs Cafe at 5pm for Isabella's party.",
            ),
            _PARTY_T1,
            _lines(
                "It's from 5 to 7pm. Isabella said everyone is welcome.",
                "There'll be music and snacks. It should be a fun way to meet people.",
                "I'm planning to go myself, so we could walk over together.",
                "No pressure, but I think it'd be fun.",
            ),
            _PARTY_T3,
        ),
        effects=_PARTY_EFFECTS,
        excluded_initiators=("isabella",),
    ),
    Topic(
        id="election",
        candidate_text="introduces himself to {b} and talks about his campaign for mayor",
        initiators=("sam",),
        requires_flag=None,
        requires_knowledge="election",
        responder_lacks="election",
        turns=(
            _lines(
                "Hello {b}! I'm Sam Moore, and I'm running for mayor in the upcoming local election.",
                "{b}, do you have a minute? I'd love to tell you about my campaign for mayor.",
                "Morning, {b}. Have you thought about who you'll vote for in the mayoral election?",
                "{b}, I'm hoping to earn your vote for mayor this spring.",
            ),
            _lines(
                "Oh, I didn't know you were running! What do you stand for?",
                "I haven't really followed local politics, to be honest.",
                "Running for mayor? That's a big step. Why now?",
                "I'm a bit busy, but go ahead.",
            ),
            _lines(
                "I want to support small businesses like Hobbs Cafe and The Willows Market and keep Johnson Park "
                "beautiful.",
                "I've lived here forty years, and I want to make sure the town stays affordable for families and "
                "students.",
                "Mainly better public transport and more support for the college community.",
                "I'd just appreciate it if you kept me in mind when you vote.",
            ),
            (
                Line("That sounds great, Sam. You have my vote.", "support"),
                Line("I like what I'm hearing. I'll tell my friends about you.", "support"),
                Line("Interesting. I'll need to think about it before I decide.", "undecided"),
                Line("I'll look into the other candidates too, but thanks for explaining.", "undecided"),
            ),
        ),
        effects={
            "support": (frozenset({"election"}), frozenset()),
            "undecided": (frozenset({"election"}), frozenset()),
        },
    ),
    Topic(
        id="small_talk",
        candidate_text="stops to chat with {b}",
        initiators=None,
        requires_flag=None,
        requires_knowledge=None,
        responder_lacks=None,
        turns=(
            _lines(
                "Hi {b}! How's your day going?",
                "Hey {b}, it's been a while. What have you been up to?",
                "Good to see you, {b}. Busy day?",
                "Oh hi {b}, nice weather today, isn't it?",
            ),
            _lines(
                "Pretty good, thanks! Just keeping busy.",
                "A bit hectic, honestly, but I'm hanging in there.",
                "Not bad at all. I've been working on a few projects.",
                "It's been a quiet one so far.",
            ),
            _lines(
                "Glad to hear it. Let's catch up properly sometime.",
                "I know the feeling. Don't forget to take a break.",
                "Sounds interesting, you'll have to tell me more later.",
                "Same here. Anyway, I should get going.",
            ),
            (
                Line("Definitely, see you around!", "none"),
                Line("Thanks, {a}. Take care!", "none"),
                Line("Will do. Have a nice day!", "none"),
                Line("Good talking to you, {a}.", "none"),
            ),
        ),
        effects={"none": (frozenset(), frozenset())},
    ),
)
TOPIC_BY_ID = {t.id: t for t in TOPICS}


def resolve_location(action: Action, persona: Persona) -> str:
    return persona.home if action.location == "home" else action.location


def spot(agent_id: str, loc_id: str) -> tuple[int, int]:
    """Buildings: everyone stands on the door tile (hidden inside); open areas: a fixed per-agent tile."""
    loc = LOC_BY_ID[loc_id]
    return loc.door if loc.door is not None else PARK_SPOTS[agent_id]


def action_allowed_for(action: Action, agent_id: str) -> bool:
    return action.agents is None or agent_id in action.agents


def hour_eligible(action: Action, hour: int) -> bool:
    return any(start <= hour < end for start, end in action.hours)


def validate() -> None:
    for a in ACTIONS:
        if a.location != "home" and a.location not in LOC_BY_ID:
            raise ValueError(f"action {a.id!r}: unknown location {a.location!r}")
        for agent_id in a.agents or ():
            if agent_id not in PERSONA_BY_ID:
                raise ValueError(f"action {a.id!r}: unknown agent {agent_id!r}")
    for p in PERSONAS:
        home = LOC_BY_ID.get(p.home)
        if home is None or home.kind != "home":
            raise ValueError(f"persona {p.id!r}: home {p.home!r} is not a home location")
    for topic in TOPICS:
        for agent_id in (topic.initiators or ()) + topic.excluded_initiators:
            if agent_id not in PERSONA_BY_ID:
                raise ValueError(f"topic {topic.id!r}: unknown initiator {agent_id!r}")
        if len(topic.turns) != 4 or any(not turn for turn in topic.turns):
            raise ValueError(f"topic {topic.id!r}: needs exactly 4 non-empty turns")
        for k, turn in enumerate(topic.turns):
            for line in turn:
                if k == 3 and line.effect not in topic.effects:
                    raise ValueError(f"topic {topic.id!r}: turn-3 line {line.text!r} has unknown effect "
                                     f"{line.effect!r}")
                if k < 3 and line.effect is not None:
                    raise ValueError(f"topic {topic.id!r}: turn-{k} line {line.text!r} must not have an effect")
    for p in PERSONAS:
        for hour in range(24):
            if not any(a.max_per_day == 99 and action_allowed_for(a, p.id) and hour_eligible(a, hour)
                       and a.requires_flag is None for a in ACTIONS):
                raise ValueError(f"persona {p.id!r}: no repeatable action available at hour {hour}")
    for loc in LOCATIONS:
        if (loc.door is None) == (loc.area is None):
            raise ValueError(f"location {loc.id!r}: needs exactly one of door/area")
        if loc.door is not None and not (0 <= loc.door[0] < MAP_W and 0 <= loc.door[1] < MAP_H):
            raise ValueError(f"location {loc.id!r}: door {loc.door} outside the {MAP_W}x{MAP_H} map")
        if loc.area is not None:
            x, y, w, h = loc.area
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > MAP_W or y + h > MAP_H:
                raise ValueError(f"location {loc.id!r}: area {loc.area} outside the {MAP_W}x{MAP_H} map")
    if set(PARK_SPOTS) != set(PERSONA_BY_ID):
        raise ValueError(f"PARK_SPOTS keys {sorted(PARK_SPOTS)} != persona ids {sorted(PERSONA_BY_ID)}")
    if len(set(PARK_SPOTS.values())) != len(PARK_SPOTS):
        raise ValueError("PARK_SPOTS tiles must be distinct")
    px, py, pw, ph = LOC_BY_ID["johnson_park"].area
    for agent_id, (x, y) in PARK_SPOTS.items():
        if not (px <= x < px + pw and py <= y < py + ph):
            raise ValueError(f"PARK_SPOTS[{agent_id!r}] = {(x, y)} is outside johnson_park area {(px, py, pw, ph)}")
    for tool in TOOLS:
        if [tech.id for tech in tool.techniques] != [f"{tool.id}_{i}" for i in range(6)]:
            raise ValueError(f"tool {tool.id!r}: needs exactly 6 techniques with ids {tool.id}_0..{tool.id}_5")
        loc = LOC_BY_ID.get(tool.location)
        if loc is None:
            raise ValueError(f"tool {tool.id!r}: unknown location {tool.location!r}")
        if (tool.tile is None) != (loc.door is not None):
            raise ValueError(f"tool {tool.id!r}: needs a tile exactly when {loc.id!r} is an open area")
        if tool.tile is not None:
            ax, ay, aw, ah = loc.area
            if not (ax <= tool.tile[0] < ax + aw and ay <= tool.tile[1] < ay + ah):
                raise ValueError(f"tool {tool.id!r}: tile {tool.tile} is outside {loc.id} area {loc.area}")
    for tool in BUILD_TOOLS:
        if [tech.id for tech in tool.techniques] != [f"{tool.id}_{i}" for i in range(6)]:
            raise ValueError(f"build tool {tool.id!r}: needs exactly 6 techniques with ids {tool.id}_0..{tool.id}_5")
        if tool.tile is not None:
            raise ValueError(f"build tool {tool.id!r}: must not have a tile")
        if tool.location != PROJECT.location:
            raise ValueError(f"build tool {tool.id!r}: location {tool.location!r} is not {PROJECT.location!r}")
    for stage in PROJECT.stages:
        tool = TOOL_BY_ID.get(stage.tool)
        if tool is None:
            raise ValueError(f"stage {stage.id!r}: unknown tool {stage.tool!r}")
        if tool.location != PROJECT.location:
            raise ValueError(f"stage {stage.id!r}: tool {stage.tool!r} is at {tool.location!r}, not "
                             f"{PROJECT.location!r}")
    project_loc = LOC_BY_ID.get(PROJECT.location)
    if project_loc is None or project_loc.area is None:
        raise ValueError(f"project {PROJECT.id!r}: location {PROJECT.location!r} is not an open area")
    ax, ay, aw, ah = project_loc.area
    for x, y in PROJECT.tiles:
        if not (ax <= x < ax + aw and ay <= y < ay + ah):
            raise ValueError(f"project {PROJECT.id!r}: tile {(x, y)} is outside {project_loc.id} area "
                             f"{project_loc.area}")
    for agent_id in PROJECT_STARTERS:
        if agent_id not in PERSONA_BY_ID:
            raise ValueError(f"PROJECT_STARTERS: unknown persona {agent_id!r}")
    if not 0 <= PROJECT.hours[0] < PROJECT.hours[1] <= 24:
        raise ValueError(f"project {PROJECT.id!r}: hours {PROJECT.hours} must satisfy 0 <= start < end <= 24")
    for a in ACTIONS:
        if a.tool is None:
            continue
        tool = TOOL_BY_ID.get(a.tool)
        if tool is None:
            raise ValueError(f"action {a.id!r}: unknown tool {a.tool!r}")
        if tool.location != a.location:
            raise ValueError(f"action {a.id!r}: tool {a.tool!r} is at {tool.location!r}, not {a.location!r}")
    for agent_id, by_tool in INITIAL_TECHNIQUES.items():
        if agent_id not in PERSONA_BY_ID:
            raise ValueError(f"INITIAL_TECHNIQUES: unknown persona {agent_id!r}")
        for tool_id, indices in by_tool.items():
            if tool_id not in TOOL_BY_ID:
                raise ValueError(f"INITIAL_TECHNIQUES[{agent_id!r}]: unknown tool {tool_id!r}")
            if any(not BASIC_TECHNIQUES <= i < 6 for i in indices):
                raise ValueError(f"INITIAL_TECHNIQUES[{agent_id!r}][{tool_id!r}]: indices {indices} must be in "
                                 f"{BASIC_TECHNIQUES}..5")
