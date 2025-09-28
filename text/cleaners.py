import re
from g2pk2 import G2p as G2p
from unidecode import unidecode
from num2words import num2words
import epitran
import cn2an
import pykakasi
import eng_to_ipa as ipa
import re

g2pj = pykakasi.kakasi()

ipa_ko = epitran.Epitran('kor-Hang')
ipa_en = epitran.Epitran('eng-Latn')
ipa_cn = epitran.Epitran('cmn-Hans', cedict_file='./cedict_1_0_ts_utf-8_mdbg.txt')
ipa_ja = epitran.Epitran('jpn-Hrgn')
ipa_ar = epitran.Epitran('ara-Arab')

ipa_es = epitran.Epitran('spa-Latn')
ipa_pt = epitran.Epitran('por-Latn')
ipa_fr = epitran.Epitran('fra-Latn')
ipa_de = epitran.Epitran('deu-Latn')
ipa_it = epitran.Epitran('ita-Latn')

ipa_fa = epitran.Epitran('fas-Arab')
ipa_tr = epitran.Epitran('tur-Latn')
ipa_vi = epitran.Epitran('vie-Latn')
ipa_id = epitran.Epitran('ind-Latn')
ipa_th = epitran.Epitran('tha-Thai')

ipa_ru = epitran.Epitran('rus-Cyrl')
ipa_hu = epitran.Epitran('hun-Latn') #
ipa_pl = epitran.Epitran('pol-Latn') #
ipa_cs = epitran.Epitran('ces-Latn') #
ipa_uk = epitran.Epitran('ukr-Cyrl') #

ipa_fi = epitran.Epitran('fin-Latn') #
ipa_sv = epitran.Epitran('swe-Latn') #
ipa_km = epitran.Epitran('khm-Khmr') #
ipa_mn = epitran.Epitran('mon-Cyrl-bab') #
ipa_ms = epitran.Epitran('mal-Mlym') #

ipa_hi = epitran.Epitran('hin-Deva') #
ipa_si = epitran.Epitran('sin-Sinh') #
ipa_ta = epitran.Epitran('tam-Taml') #
ipa_nl = epitran.Epitran('nld-Latn') #
ipa_te = epitran.Epitran('tel-Telu') #

ipa_ur = epitran.Epitran('urd-Arab') #
ipa_fil = epitran.Epitran('tgl-Latn')

#ipa_bn = epitran.Epitran('ben-Beng') #
_whitespace_re = re.compile(r'\s+')
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    # - For replication of https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2
    # you may need to replace the symbol to Russian one
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = text.lower()
    text = collapse_whitespace(text)
    return text


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i:i+3] == 'ㅇㅡㄹ' or text[i:i+3] == 'ㄹㅡㄹ') and text[i+3] == ' ' and text[i+4] == 'ㄹ':
            new_text += text[i:i+3] + ' ' + 'ㄴ'
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text

# Function to extract all the numbers from the given string
def numCleaner(str, lang):
	nums = re.findall(r'[0-9]+[.]?[0-9]*', str) #[-+]?
	for num in nums:
		if "." in num:
			val = float(num)
		else:
			val = int(num)
		if lang != 'cn':
			str = str.replace(num, num2words(val, lang=lang))
		else:
			str = str.replace(num, cn2an.an2cn(num))
	return str

def canvers_en_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'en')
    phonemes = ipa.convert(text) #ipa_en.transliterate(text) #
    return collapse_whitespace(phonemes)

def canvers_ja_cleaners(text):
    text = expand_abbreviations(text.lower())
    result = g2pj.convert(numCleaner(text,'ja'))
    text = ""
    for item in result:
        text = text + item['hira']
    phonemes = ipa_ja.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_zh_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'cn')
    phonemes = ipa_cn.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_ko_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'ko')
    phonemes = ipa_ko.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_ar_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'ar')
    phonemes = ipa_ar.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_es_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'es')
    phonemes = ipa_es.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_pt_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'pt')
    phonemes = ipa_pt.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_fr_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'fr')
    phonemes = ipa_fr.transliterate(text)
    return collapse_whitespace(phonemes)



def canvers_vi_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'vi')
    phonemes = ipa_vi.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_id_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'id')
    phonemes = ipa_id.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_th_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'th')
    phonemes = ipa_th.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_ru_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'ru')
    phonemes = ipa_ru.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_fa_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'fa')
    phonemes = ipa_fa.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_tr_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'tr')
    phonemes = ipa_tr.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_de_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'de')
    phonemes = ipa_de.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_it_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'it')
    phonemes = ipa_it.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_cs_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'cz')
    phonemes = ipa_cs.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_pl_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'pl')
    phonemes = ipa_pl.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_hu_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'hu')
    phonemes = ipa_hu.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_uk_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'uk')
    phonemes = ipa_uk.transliterate(text)
    return collapse_whitespace(phonemes)
 
def canvers_fi_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'fi')
    phonemes = ipa_fi.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_sv_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'sv')
    phonemes = ipa_sv.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_km_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'fi')
    phonemes = ipa_km.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_mn_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_mn.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_ms_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'fi')
    phonemes = ipa_ms.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_hi_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_hi.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_si_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_si.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_ta_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_ta.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_nl_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'nl')
    phonemes = ipa_nl.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_te_cleaners(text):
    text = expand_abbreviations(text.lower())
    text = numCleaner(text,'te')
    phonemes = ipa_te.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_ur_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_ur.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_fil_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_fil.transliterate(text)
    return collapse_whitespace(phonemes)

"""
def canvers_bn_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_bn.transliterate(text)
    return collapse_whitespace(phonemes)


def canvers_hk_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_hk.transliterate(text)
    return collapse_whitespace(phonemes)

def canvers_he_cleaners(text):
    text = expand_abbreviations(text.lower())
    #text = numCleaner(text,'sv')
    phonemes = ipa_he.transliterate(text)
    return collapse_whitespace(phonemes)
"""