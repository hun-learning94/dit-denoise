"""Presentation notation; historical P/G metric keys remain unchanged."""
import re

SYMBOLS = {'P': 'EA', 'G': 'SA', 'F': 'Agr', 'S': 'JNA'}
_TOKEN = re.compile(r'(?<![A-Za-z\\])([PGFS])(?![A-Za-z])')
_PLAIN = re.compile(r'(?<![\w\\])([PGFS])(?!\w)')


def math_notation(text):
    def replace(match):
        value = r'\mathrm{' + SYMBOLS[match[1]] + '}'
        # A multi-letter correlation/contrast subscript needs its own group.
        return '{' + value + '}' if match.start() and text[match.start()-1] in '_^' else value
    return _TOKEN.sub(replace, text)


def display_text(text):
    """Translate rendered prose and dollar-delimited math, never data keys."""
    parts = re.split(r'(\$\$.*?\$\$|\$[^$\n]*\$)', text, flags=re.S)
    return ''.join(math_notation(part) if part.startswith('$') else
                   _PLAIN.sub(lambda m: SYMBOLS[m[1]], part) for part in parts)


def style_figure(fig):
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        artist.set_text(display_text(artist.get_text()))
