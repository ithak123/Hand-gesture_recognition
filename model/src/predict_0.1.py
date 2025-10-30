import torch
import torch.nn.functional as F

from model.src.model_architecture import GestureCNN
from model.src.dataset_loader import prepare_data
from model.config.config import BASE_PATH, IMG_SIZE, BATCH_SIZE,NUM_WORKERS


#דקרטור שבא להבטיח של נחשב גרדיאנטים
@torch.no_grad()
def predict_batch(model: GestureCNN, images: torch.Tensor, class_names: list[str], topk: int = 3 ):

    model.eval() #אנחנו מכניסים את המודל למצב בחינה כך שהוא לא עושה DROPOUT וכו'
    logits = model(images) #בעצם כאן אנחנו מעבירים את כל התמונות בבאצ' במודל
    probas = F.softmax(logits, dim=1) #מחזיר לנו את רשימת החיזויים לאחר פונקציית ההפעלה

    #יprobas.topk מחזיר שני טנזורים אחד של הערכים עצמם(הסתברויות), והשני האינדקסים(המספרים של החלקות המובילות)
    top_probs, top_idx = probas.topk(k=topk, dim=1)

    #מחחזיר לנו רשימה פייתונית של שמות המחלקות שנחזו
    topk_labels = [[class_names[j] for j in row.tolist()] for row in top_idx]

    #ממיר את ההסתברויות לרשימה פייונית
    top_probs = [row.tolist() for row in top_probs]

    # 📦 מחזיר שלישייה:
    # logits – הציונים הגולמיים
    # probs – ההסתברויות לכל מחלקה
    # topk – רשימת זוגות (labels, probs) לכל תמונה
    # zip משדך בין שתי הרשימות כדי שכל תמונה תקבל רשימת (שם מחלקה + הסתברות)
    return logits, probas, list(zip(topk_labels, top_probs))


def main():

    #טעינת שלושת הדאטה לאודרים
    train_loader, val_loader, test_loader = prepare_data(base_path=BASE_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS )

    #שומר את שמות המחלקות
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    print(f" class_names: {class_names} (num_classes={num_classes})")

    #יוצר מופע של המודל
    model = GestureCNN(num_classes=num_classes)

    # לוקח באצ' אחד(ראשון) מתוך Test set
    images, labels = next(iter(test_loader))
    # הדפסת המידות שלמ הטנזורים לצורך בקרה
    print(f" batch images shape: {images.shape} | labels shape: {labels.shape}")

    #הרצה של החיזוי בפועל
    #מחזיר ציונים גולמיים, הסתברויות , ורשימה של 3 המחלקות המובילות לכל תמונה
    logits, probas, topk = predict_batch(model, images, class_names, topk=3)


    # רץ על 5 הדוגמאות הראשונות בבאצ' (או פחות אם יש פחות מ-5 תמונות) לצורך הצגתם כדוגמה
    for i in range(min(5, images.size(0))):

       # מוציאים את התווית האמתית של התמונה בתור מספר פייתוני
       true_label = class_names[labels[i].item()]
       #יTuple של (labels, probs)
       pred_labels, pred_probs = topk[i]
       pred_str = ", ".join([f"{lbl}: {p:.3f}" for lbl, p in zip(pred_labels, pred_probs)])
       print(f"[{i}] GT={true_label} | TOP3 => {pred_str}")


if __name__ == '__main__':
    main()




