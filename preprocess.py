from PIL import Image,ImageOps, ImageChops
import os, sys


def convert(source='dataset',target='converted'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im=Image.open(read+file)
        f, e = os.path.splitext(read+file)
        outfile = f[f.rfind('\\')+1:f.__len__()] + ".jpg"
        if e=='.gif':
            i = 0
            mypalette = im.getpalette()
            try:
                while 1:
                    im.putpalette(mypalette)
                    new_im = Image.new("RGBA", im.size)
                    new_im.paste(im)
                    background = Image.new("RGB", new_im.size, (255, 255, 255))
                    background.paste(new_im, mask=new_im.split()[3]) # 3 is the alpha channel
                    background.save(write+outfile)
                    i += 1
                    im.seek(im.tell() + 1)
            except EOFError:
                continue
        else:
            try:
                im=im.convert('RGB')
                im.save(write+outfile)
            except IOError:
                print("cannot convert", read+file)

def resize(width=150,height=150,source='converted',target='resized'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=im.resize((width,height))
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)
            
def grayscale(source='converted',target='black'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=im.convert("L")
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)
            
def trim1(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def trim2(im):
    width, height = im.size
    bg = Image.new(im.mode, im.size, im.getpixel((width-1,height-1)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def croping(source='converted',target='croped'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=trim1(im)
            im=trim2(im)
            """im=im.convert('RGB')
            bbox=im.getbbox()
            im=im.crop(bbox)
            invert_im = ImageOps.invert(im)
            wbox=invert_im.getbbox()
            im=im.crop(wbox)"""
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)   