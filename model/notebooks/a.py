from dataset_loader import prepare_data
import time
import multiprocessing
multiprocessing.freeze_support()

if __name__ == "__main__":
    BASE_PATH = r"/model/data/data_split"

    for workers in [0, 2, 4]:
        print(f"\n=== בדיקה עם num_workers={workers} ===")
        t0 = time.time()
        train_loader, val_loader, test_loader = prepare_data(BASE_PATH, num_workers=workers)

        for i, (imgs, lbls) in enumerate(train_loader):
            if i >= 20: break  # רק לבדיקה מהירה

        print(f"זמן טעינה: {time.time() - t0:.2f} שניות")
