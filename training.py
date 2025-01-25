import pandas as pd
import tensorflow as tf
import tf_keras as keras
from constants import (PROCESSED_DATA_DIR,
                       METADATA_FILEPATH,
                       BATCH_SIZE,
                       EPOCHS,
                       BERT_BASE,
                       MAX_SEQUENCE_LENGHT,
                       FilePath,
                       PageMetadata,
                       ImageSize,
                       ImageInputShape)
from pandera.typing import DataFrame
from typing import Tuple, List
from transformers import TFBertModel
from tf_keras import layers, models
from PIL import Image

# Allow for unlimited image size, some documents are pretty big...
Image.MAX_IMAGE_PIXELS = None


def stratified_split(
        df: pd.DataFrame,
        train_frac: float,
        val_frac: float,
        test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dfs, val_dfs, test_dfs = [], [], []

    for label, group in df.groupby('label'):
        n = len(group)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        train_dfs.append(group.iloc[:train_end])
        val_dfs.append(group.iloc[train_end:val_end])
        test_dfs.append(group.iloc[val_end:])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    return train_df, val_df, test_df


def dataset_from_dataframe(df: pd.DataFrame) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((
        df['img_filepath'].values,
        df['input_ids'].values,
        df['attention_mask'].values,
        df['label'].values,
    ))


def load_image(image_path: FilePath, image_size: ImageSize) -> Image:
    img_width, img_height = image_size

    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_width, img_height])
    image /= 255.0

    return image


def prepare_dataset(
        ds: tf.data.Dataset,
        image_size: ImageSize,
        batch_size=32,
        buffer_size=1000
) -> tf.data.Dataset:
    def load_image_and_format_tensor_shape(
            img_path: FilePath,
            input_ids: List[int],
            attention_mask: List[int],
            label: str
    ):
        image = load_image(img_path, image_size)
        return ((image, input_ids, attention_mask), label)

    return ds.map(
        load_image_and_format_tensor_shape,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ) \
        .shuffle(buffer_size=buffer_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)


metadata_df: DataFrame[PageMetadata] = pd.read_csv(METADATA_FILEPATH)
metadata_df = metadata_df.sample(n=50, random_state=42)

median_height = int(metadata_df['height'].median())
median_width = int(metadata_df['width'].median())
img_size: ImageSize = (median_height, median_width)
img_input_shape: ImageInputShape = img_size + (3,)

label_names: List[str] = sorted(
    [d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()]
)
num_classes = len(label_names)

print('Splitting the DataFrame into training, validation and test')
train_df, val_df, test_df = stratified_split(
    metadata_df,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
)

print('Batching and shuffling the datasets')
train_ds = dataset_from_dataframe(train_df)
train_ds = prepare_dataset(train_ds, img_size, batch_size=BATCH_SIZE)

val_ds = dataset_from_dataframe(val_df)
val_ds = prepare_dataset(val_ds, img_size, batch_size=BATCH_SIZE)

test_ds = dataset_from_dataframe(test_df)
test_ds = prepare_dataset(test_ds, img_size, batch_size=BATCH_SIZE)


def build_image_model(input_shape: ImageInputShape) -> keras.Model:
    img_model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
    ], name='image_classification')

    img_model.summary()
    return img_model


def build_text_model() -> keras.Model:
    bert_model = TFBertModel.from_pretrained(BERT_BASE)

    input_ids = layers.Input(
        shape=(MAX_SEQUENCE_LENGHT,), dtype=tf.int32, name='input_ids'
    )
    attention_mask = layers.Input(
        shape=(MAX_SEQUENCE_LENGHT,), dtype=tf.int32, name='attention_mask'
    )

    # The second element of the BERT output is the pooled output i.e. the
    # representation of the [CLS] token
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]

    text_model = models.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs,
        name='bert'
    )
    text_model.summary()

    return text_model


def build_multimodal_model(
        num_classes: int,
        img_input_shape: ImageInputShape
) -> keras.Model:
    img_model = build_image_model(img_input_shape)
    text_model = build_text_model()

    img_input = layers.Input(shape=img_input_shape, name='img_input')
    text_input_ids = layers.Input(
        shape=(MAX_SEQUENCE_LENGHT,), dtype=tf.int32, name='text_input_ids'
    )
    text_input_mask = layers.Input(
        shape=(MAX_SEQUENCE_LENGHT,), dtype=tf.int32, name='text_input_mask'
    )

    img_features = img_model(img_input)
    text_features = text_model([text_input_ids, text_input_mask])

    classification_layers = keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='classification_layers')
    concat_features = layers.concatenate([img_features, text_features],
                                         name='concatenate_features')
    outputs = classification_layers(concat_features)

    multimodal_model = models.Model(
        inputs=[img_input, text_input_ids, text_input_mask],
        outputs=outputs,
        name='multimodal_document_page_classifier'
    )
    return multimodal_model


multimodal_model = build_multimodal_model(num_classes, img_input_shape)
multimodal_model.summary()
multimodal_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
multimodal_model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
)
