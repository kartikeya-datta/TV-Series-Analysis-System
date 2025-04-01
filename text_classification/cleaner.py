from bs4 import BeautifulSoup
class cleaner():
    def __init__(self):
        pass
    
    def put_line_breaks(self, text):
        return text.replace("<\p>", "<\p>\n")
    
    def remove_html_tag(self, text):
        clean_text = BeautifulSoup(text, "html.parser").get_text()
        return clean_text
    
    def clean_text(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tag(text)
        text = text.strip()
        return text