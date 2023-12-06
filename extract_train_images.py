import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np


# Function to rotate an image
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result


# Function to process each bounding box in the XML data
def process_box(box, image, output_folder, count):
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


# Function to process the XML file
def process_xml(xml_path, image, output_folder):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = {}
    # Iterate over each box in the XML data
    for box in root.findall('.//box'):
        new_count = process_box(box, image, output_folder, count)
        count = new_count

if __name__ == '__main__':
    # Set the path to the XML file and the folder containing images
    train_xml_path = 'route_annotations.xml'
    station_xml_path = 'city_annotations.xml'
    image_path = 'cropped_board_images/cropped_messy2_stations.jpg'
    #train_output = 'messy2_trains_in_some_spots'
    station_output = 'messy2_stations_in_some_spots'

    # Read the image
    image = cv2.imread(image_path)
    desired_height, desired_width, channels = 2388, 3582, 3
    if image.shape != (desired_height, desired_width, channels):
        image = cv2.resize(image, (desired_width, desired_height))

    process_xml(station_xml_path, image, station_output)
    #process_xml(train_xml_path, image, train_output)

    
