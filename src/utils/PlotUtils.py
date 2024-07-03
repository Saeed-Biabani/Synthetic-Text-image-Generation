from PIL import ImageFont, ImageDraw, Image
import numpy as np

class PlotPred:
    def __init__(self, font = "../../../Desktop/Roshd/OCR/DataGeneration/Fonts/IRANSansWeb.ttf"):
        self.font_ = ImageFont.truetype(font, 25)
        
    
    def generate_base(self, shape):
        return np.ones(shape + (3, ), dtype = "uint8") * 255
    
    def place_image_in_bg(self, img, bg):
        offset = (np.array(bg.shape) - np.array(img.shape))[:2] // 2
        bg[int(offset[0]*1.5):int(offset[0]*1.5)+img.shape[0], offset[1]:offset[1]+img.shape[1], :] = img
        return bg
    
    def write_text(self, img, text):
        h, w, _ = img.shape

        img_pil = Image.fromarray(img)

        draw = ImageDraw.Draw(img_pil)

        draw.text((w/2, h/10 + 1.25*40), text = text,
                font = self.font_, fill = 0,
                language = "fa", anchor = "md",
                align = 'center', direction = 'rtl')

        return img_pil
    
    def plot(self, img, title):
        bg = self.generate_base(tuple(np.array(np.array(img.shape[:2])*np.array((2.75, 1.35)), dtype = "int")))
        mix = self.place_image_in_bg(img, bg)
        
        return self.write_text(mix, title)