import pandas as pd
import re
from janome.tokenizer import Tokenizer

# Initialisation du tokenizer Janome
tokenizer = Tokenizer()

def clean_text(text):
    """
    Enlève les caractères non imprimables et invisibles problématiques.
    """
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def tokenize_japanese(text, max_length=5000):
    """
    Tokenize un texte japonais en une liste de tokens.
    Gère les entrées vides, non-string, ou trop longues.
    """
    if not isinstance(text, str):
        return []
    text = clean_text(text.strip())
    if text == "":
        return []
    if len(text) > max_length:
        text = text[:max_length]

    try:
        tokens = tokenizer.tokenize(text)
        return [token.surface for token in tokens]
    except Exception as e:
        print(f"[Tokenization Error] {e} — text was: {text[:30]}...")
        return []

def apply_tokenization(df):
    """
    Applique la tokenization à la colonne 'text' d'un DataFrame.
    Retourne une copie du DataFrame avec une nouvelle colonne 'tokens'.
    """
    df = df.copy()
    df['text'] = df['text'].fillna('').astype(str)  # Remplacer NaN et forcer str
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de DataFrame
    data = {
        'text': [
            "これはテストです。",
            None,
            "",
            "  ",
            "非常に長いテキスト" * 1000,  # Texte très long
            "特殊文字\u0000\u001fを含むテキスト"
        ]
    }
    df = pd.DataFrame(data)
    df = apply_tokenization(df)
    print(df[['text', 'tokens']])
