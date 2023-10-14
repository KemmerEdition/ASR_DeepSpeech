# Don't forget to support cases when target_text == ''

import editdistance

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    if not target_text:
        if predicted_text:
            return 1
        return 0
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    if not target_text:
        if predicted_text:
            return 1
        return 0
    target_text_splited = target_text.split(' ')
    predicted_text_splited = predicted_text.split(' ')
    return editdistance.distance(target_text_splited, predicted_text_splited) / len(target_text_splited)
