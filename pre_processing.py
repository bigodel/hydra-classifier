# -*- coding: utf-8 -*-
import os
import pandas as pd
import pdf2image as p2i
import pytesseract
from os import path
from PIL import Image
from typing import List, Tuple
from transformers import BertTokenizer
from constants import (RAW_DATA_DIR,
                       PROCESSED_DATA_DIR,
                       METADATA_FILEPATH,
                       BERT_BASE,
                       MAX_SEQUENCE_LENGHT,
                       FilePath,
                       PageMetadata)

# Allow for unlimited image size, some documents are pretty big...
Image.MAX_IMAGE_PIXELS = None


def make_page_filepaths(basename, label, page_index) -> Tuple[str, str]:
    out_dirname = path.join(PROCESSED_DATA_DIR, label)
    os.makedirs(out_dirname, exist_ok=True)
    out_filename = path.join(out_dirname, f'{basename}_{page_index}')

    out_img_filepath = f'{out_filename}.jpg'
    out_txt_filepath = f'{out_filename}.txt'

    return out_img_filepath, out_txt_filepath


def tokenize_text(text: str) -> Tuple[List[int], List[int]]:
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE)
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGHT,
    )

    return tokenized['input_ids'], tokenized['attention_mask']


def process_pdf_file(pdf_filepath: FilePath):
    if path.getsize(pdf_filepath) == 0:
        # TODO: substitute for logging
        print(f'{pdf_filepath} is empty, skipping')
        return []

    pages: List[Image] = p2i.convert_from_path(pdf_filepath)
    pages_metadata: List[PageMetadata] = []

    root_dir, doctype = path.split(path.dirname(pdf_filepath))
    base_filename = path.basename(path.splitext(pdf_filepath)[0])
    for page_i, page in enumerate(pages):
        label = 'other'
        if page_i == 0:
            label = doctype
        # If the document only has one page, override the label with
        if page_i == len(pages) - 1:
            label = f'{doctype}-last'

        out_img_filepath, out_txt_filepath = make_page_filepaths(base_filename, label, page_i)

        page.save(out_img_filepath)

        ocr_text = pytesseract.image_to_string(page)
        input_ids, attention_mask = tokenize_text(ocr_text)
        with open(out_txt_filepath, 'w') as out_txt_file:
            out_txt_file.write(ocr_text)

        pages_metadata.append({
            'page_number': page_i + 1,
            'pdf_filepath': path.relpath(pdf_filepath, start='.'),
            'img_filepath': out_img_filepath,
            'txt_filepath': out_txt_filepath,
            # 'text_tokens': tokens,
            'width': page.width,
            'height': page.height,
            'label': label,
        })

    return pages_metadata


def process_training_data() -> pd.DataFrame:
    pages_metadata: List[List[PageMetadata]] = []

    for dirname, _, files in os.walk(RAW_DATA_DIR):
        if path.samefile(dirname, RAW_DATA_DIR):
            continue

        print(f'Processing folder {dirname}')

        for filename in files:
            _, ext = path.splitext(filename)

            # Avoid processing non-document files
            if ext.lower() == '.pdf':
                print(f'Processing file {filename}')
                pdf_filepath = path.join(dirname, filename)
                pages_metadata.extend(process_pdf_file(pdf_filepath))

    pages_metadata_df = pd.DataFrame(pages_metadata)
    print(f'Writing metadata to {METADATA_FILEPATH}')
    pages_metadata_df.to_csv(METADATA_FILEPATH, index=False)
    return pages_metadata_df


process_training_data()
