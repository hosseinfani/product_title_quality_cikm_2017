from enum import Enum
#http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
class Pos(Enum):
    CoordinatingConjunction = 'CC'
    CardinalNumber = 'CD'
    Determiner = 'DT'
    ExistentialThere = 'EX'
    ForeignWord = 'FW'
    PrepositionOrSubordinatingConjunction = 'IN'
    Adjective = 'JJ'
    AdjectiveComparative = 'JJR'
    AdjectiveSuperlative = 'JJS'
    ListItemMarker = 'LS'
    Modal = 'MD'
    NounSingularOrMass = 'NN'
    NounPlural = 'NNS'
    ProperNounSingular = 'NNP'
    ProperNounPlural = 'NNPS'
    Predeterminer = 'PDT'
    PossessiveEnding = 'POS'
    PersonalPronoun = 'PRP'
    PossessivePronoun = 'PRP$'
    Adverb = 'RB'
    AdverbComparative = 'RBR'
    AdverbSuperlative = 'RBS'
    Particle = 'RP'
    Symbol = 'SYM'
    to = 'TO'
    Interjection = 'UH'
    VerbBaseForm = 'VB'
    VerbPastTense = 'VBD'
    VerbGerundOrPresentParticiple = 'VBG'
    VerbPastParticiple = 'VBN'
    VerbNon3rdPersonSingularPresent = 'VBP'
    Verb3rdPersonSingularPresent = 'VBZ'
    WhDeterminer = 'WDT'
    WhPronoun = 'WP'
    PossessiveWhPronoun = 'WP$'
    WhAdverb = 'WRB'
    Punctuation = 'PUNCT'
