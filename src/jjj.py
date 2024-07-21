import pymupdf
from PIL import Image, ImageDraw
import numpy as np

def create_mask(image_size, points):
    mask = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = np.array(mask)
    return mask

def apply_mask(image, mask):
    masked_image = Image.fromarray(np.array(image) * mask[:, :, np.newaxis])
    return masked_image

def extract_regions_from_pdf(pdf_path, annotation, output_dir):
    # Open the PDF file
    pdf_document = pymupdf.open(pdf_path)
    
    # Extract the first page
    page = pdf_document.load_page(0)
    
    # Convert the page to a PIL image
    pix = page.get_pixmap()
    img = pix.tobytes()
    #mode = "RGBA" if pix.alpha else "RGB"
    #page_image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    
    # Process each box
    for idx, box in enumerate(annotation['boxes']):
        points = [(int(x), int(y)) for x, y in box['points']]
        
        # Create the mask
        mask = create_mask(img.size, points)
        
        # Apply the mask to the image
        cutout_image = apply_mask(img, mask)
        
        # Crop the image to the bounding box
        x = int(float(box['x']))
        y = int(float(box['y']))
        width = int(float(box['width']))
        height = int(float(box['height']))
        cutout_image = cutout_image.crop((x, y, x + width, y + height))
        
        # Save the cutout image
        output_path = f"{output_dir}/cutout_{idx}.png"
        cutout_image.save(output_path)
        print(f"Saved cutout image to {output_path}")

# Annotation and PDF path
annotation = {
    "boxes": [
        {
            "type": "polygon",
            "label": "ww",
            "x": "979.4941",
            "y": "1126.7607",
            "width": "342.4805",
            "height": "910.9980",
            "points": [
                [818.5283203125, 684.9609375],
                [808.25390625, 695.2353515625],
                [808.25390625, 784.2802734375],
                [842.501953125, 1571.9853515625],
                [852.7763671875, 1582.259765625],
                [893.8740234375, 1582.259765625],
                [1140.4599609375, 1568.560546875],
                [1150.734375, 1558.2861328125],
                [1150.734375, 1496.6396484375],
                [1137.03515625, 1037.7158203125],
                [1116.486328125, 681.5361328125],
                [1106.2119140625, 671.26171875],
                [1037.7158203125, 671.26171875],
                [873.3251953125, 678.111328125]
            ]
        },
        # Add other boxes here...
    ],
    "height": 2480,
    "key": "img5.jpg",
    "width": 3507
}
pdf_path = "data\\Chão 350 Arv2.pdf"
output_dir = "data"

# Extract and save regions
extract_regions_from_pdf(pdf_path, annotation, output_dir)