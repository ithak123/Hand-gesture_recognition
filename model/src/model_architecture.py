import torch
import torch.nn as nn
import torch.nn.functional as F # גרסה פונקציונאלית (בשבלי RELU וכו' שלא מוגדים כSELF )


#יnn.Module נותן לנו את כל הכונות החשובות
class GestureCNN(nn.Module):

    def __init__(self, num_classes=10): #הבנאי של המחלקה ליצירת מודל
        super(GestureCNN, self).__init__() #קורא לבנאי של nn.Module כדי לאתחל את המשקלים והמעקב


        # --- Conv Block 1 ---
        #יpadding בכדי שנוכל לספור גם את הנוירונים בקצוות
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        #אנחנו מכניסים את הpool לבנאי בשביל הנוחות
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #הDropout נמצא ההנאי כי כשאנחנו נמצאים בשלב הEVAL (לא אימון)
        self.dropout1 =  nn.Dropout(p=0.25)


        # --- Conv Block 2 ---
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=0.25)


        # --- Conv Block 2 ---
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(p=0.25)


        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        #בחלק הזה יש לנו הכי הרבה משקלים אז אנחנו עושים דרופ חזק בשביל למנוע אוברפיט(עשינו שלוש שכבות וזה גם ככה סיכון)
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        # X = הטנזור שלנו

        # --- Block 1 ---
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # --- Block 2 ---
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        # --- Block 3 ---
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout3(x)

        # --- Flatten ---
        #מקבל טנזור  לדוגמה: [batch, 64, 12, 12] -> [batch, 9216]
        x = torch.flatten(x, 1)

        # --- Fully Connected ---
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)


        return x # מחזיר טנזור [batch, 10] ומכאן שולחים את זה לפונקציית חיזוי



if __name__ == "__main__":
    model = GestureCNN(num_classes=10)
    x = torch.randn(1, 1, 96, 96)  # תמונה אחת לדוגמה
    out = model(x)

    print(f"✅ צורת הקלט: {x.shape}")
    print(f"✅ צורת הפלט (logits): {out.shape}")
    print(f"✅ סך כל פרמטרים לאימון: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")