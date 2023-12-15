import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np


def rotate_image(image, angle):
    """
    Rotate the input image by a specified angle.

    Args:
    - image (numpy.ndarray): Input image.
    - angle (float): Rotation angle in degrees.

    Returns:
    - numpy.ndarray: Rotated image.
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result


def process_box(box, image, output_folder, count):
    """
    Process a bounding box in XML data, crop, rotate, and save the corresponding region of the image.

    Args:
    - box (Element): Bounding box element in XML data.
    - image (numpy.ndarray): Original image.
    - output_folder (str): Output folder to save processed images.
    - count (dict): Dictionary to keep track of label counts.

    Returns:
    - dict: Updated label count dictionary.
    """
    label = box.attrib['label']
    xtl, ytl, xbr, ybr = map(float, [box.attrib['xtl'], box.attrib['ytl'], box.attrib['xbr'], box.attrib['ybr']])
    
    pad = 40
    if 'rotation' in box.attrib:
        rotation = float(box.attrib.get('rotation', 0))
    else:
        rotation = 0

    # Crop the region defined by the bounding box
    # Padding to capture parts of the image that may be rotated in
    cropped_image = image[int(ytl - pad):int(ybr + pad), int(xtl):int(xbr)]

    # Rotate the cropped image if rotation is specified
    if rotation != 0:
        cropped_image = rotate_image(cropped_image, rotation)

    trimmed_image = cropped_image[pad:-pad, :]
    # Save the cropped and rotated image to the output folder
    if label not in count:
        count[label] = 1
    else:
        count[label] += 1
    output_path = os.path.join(output_folder, f'{label}-{count[label]}.jpg')
    cv2.imwrite(output_path, trimmed_image)
    return count


def process_xml(xml_path, image, output_folder):
    """
    Process an XML file containing bounding box annotations and save cropped and rotated images.

    Args:
    - xml_path (str): Path to the XML file.
    - image (numpy.ndarray): Original image.
    - output_folder (str): Output folder to save processed images.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = {}
    # Iterate over each box in the XML data
    for box in root.findall('.//box'):
        new_count = process_box(box, image, output_folder, count)
        count = new_count

def extract_images(image_path, station_output, train_output):
    """
    Extract images from an input image based on bounding box annotations in XML files.

    Args:
    - image_path (str): Path to the input image.
    - station_output (str): Output folder for processed station images.
    - train_output (str): Output folder for processed train images.
    """
    train_xml_path = 'route_annotations.xml'
    station_xml_path = 'city_annotations.xml'

    # Read the image
    image = cv2.imread(image_path)
    desired_height, desired_width, channels = 2388, 3582, 3
    if image.shape != (desired_height, desired_width, channels):
        image = cv2.resize(image, (desired_width, desired_height))

    process_xml(station_xml_path, image, station_output)
    process_xml(train_xml_path, image, train_output)


if __name__ == '__main__':
    extract_images()
    
