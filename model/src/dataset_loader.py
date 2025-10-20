from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, random


def get_transforms(img_size=96):

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),# אנחנו רוצים ערוץ אחד
        transforms.Resize((img_size, img_size)),# Tuple, בכדי לשמור על התמונה ריבועית בכדי לשמור על דאטה אחיד למודל
        transforms.ToTensor(),# מטריצה תלת מימדית של פיקסלים(אצלי המימד השליש הוא של שכבה אחת של בהירות)
        # לא מוסיפים Normalize — נשאר בטווח [0,1]
    ])


def laod_dataset(base_path, img_size=96):

    transform = get_transforms(img_size)

    # בדיקת קיימות של הדאטה
    for subset in ["train", "val", "test"]:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f" לא נמצאה התיקייה: {subset_path}")


    #יצירת אובייקטים של ImageFolder (יודע לדלות את הלייבלים מהתיקיות),ונומליזציה לדאטה(COMPOSE)
    train_dataset = datasets.ImageFolder(os.path.join(base_path, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(base_path, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_path, "test"), transform=transform)

    #ערבוב הדאטה של test ו val כדי שהמודל לא יבחן על התמונות מסודרות בינהים
    random.seed(42) # ראנדום קבוע
    for ds_name, ds in [("val", val_dataset), ("test", test_dataset)]: # List of Tupel's

        idx = list(range(len(ds))) #רשימה של כל האינדקסים של התמונות בדאסהט הספציפי
        random.shuffle(idx)#ערבוב של רשימת האינדקסים

        #יsamples היא רשימה שהImageFolder יצר והיא מחזיקה את הנתיב לתמונה ואת הלייבל שלה
        ds.samples = [ds.samples[i] for i in idx]#זו השורה שעושה את הערבוב בפועל
        print(f" בוצע ערבוב קבוע לסט {ds_name} ({len(ds)} תמונות)")


    print(f"✅ train: {len(train_dataset)} תמונות")
    print(f"✅ val:   {len(val_dataset)} תמונות (מעורבבות)")
    print(f"✅ test:  {len(test_dataset)} תמונות (מעורבבות)")

    return train_dataset, val_dataset, test_dataset


#יצירת 3 דאטה לאודרים (מנהלים את הדאטה לאימון)
def get_dataloaders(train_ds, val_ds, test_ds, batch_size=32, num_workers=0):

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


#ונקציית הMAIN
def prepare_data(base_path, img_size=96, batch_size=32, num_workers=0):

    train_ds, val_ds, test_ds = laod_dataset(base_path, img_size=img_size)
    return get_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    BASE_PATH =r"C:\Users\ithak\PycharmProjects\Hand-gesture_recognition\model\data\data_split"
    train_loader, val_loader, test_loader = prepare_data(BASE_PATH)

    # דוגמה לבדיקה: קבלת באצ' ראשון
    images, labels = next(iter(train_loader))
    print(f"\n🖼️ צורת התמונות בבאצ' הראשון: {images.shape}")
    print(f"🏷️ תגים לדוגמה: {labels[:8].tolist()}")