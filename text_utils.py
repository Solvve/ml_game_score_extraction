import pytesseract

def extract_text(image, lang = 'eng'):
    config = r'--psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=" 01234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm-:"'
    
    return pytesseract.image_to_string(image, config=config, lang=lang)

def parse_score(score_text):
    score_line = score_text.split('\n')[-1]
    ret = {
        "score": score_line,
        "team1": None, 
        "score1": None,
        "score2": None,
        "team2": None
    }
    
    results = score_line.split(' - ')
    if len(results) != 2: return ret
    
    result1 = results[0].split(' ')
    ret["team1"] = " ".join(result1[:-1])
    ret["score1"] = result1[-1]

    result2 = results[1].split(' ')
    ret["team2"] = " ".join(result2[1:])
    ret["score2"] = result2[0]
    
    return ret

def extract_score(image, lang = 'eng'):
    text = extract_text(image, lang)
    text = text.replace("\n\x0c", "")

    print(f'Extracted text:\n{text}')
    
    return parse_score(text)