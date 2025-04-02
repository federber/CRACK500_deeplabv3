from utils import *


dataset_path_dir = '/app/CRACK500'
save_folder = '/app/results'



encoder_weights = 'imagenet'
activation = None

model = smp.DeepLabV3(
    encoder_weights=encoder_weights,
    classes=2,
    activation=activation,
)

#если начинаем не с начала
#checkpoint_path = "/home/jupyter/datasphere/project/results/crack_dlv3_model.pt"
#model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

#Запускаем обучение
trainer = DeepLabv3_trainer(model, dataset_path_dir)
trainer.train_model(save_folder)

#Визуализация результатов
resultsDF = pd.read_csv(Path(save_folder) / "results_crack.csv")

plt.figure(figsize=(10, 10))
plt.plot(resultsDF['epoch'], resultsDF['training_loss'], label="Training Loss")
plt.plot(resultsDF['epoch'], resultsDF['val_loss'], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 10))
plt.plot(resultsDF['epoch'], resultsDF['training_f1'], label="Training F1-score")
plt.plot(resultsDF['epoch'], resultsDF['val_f1'], label="Validation F1-score")

plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("Training and Validation F1-score")
plt.legend()
plt.grid()
plt.show()

#Запускаем на тестовых данных
device = 'cuda'  
encoder_weights = 'imagenet'
activation = None

model = smp.DeepLabV3(
    encoder_weights=encoder_weights, 
    classes=2, 
    activation=activation,
).to(device)

checkpoint_path = "/home/jupyter/datasphere/project/results/crack_dlv3_model.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

test_ds = SegData(dataset_path=dataset_dir_path, file_name='test.txt', transform=None)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

acc = tm.Accuracy(task="multiclass", average="micro", num_classes=2).to(device)
f1 = tm.F1Score(task="multiclass", average="macro", num_classes=2).to(device)
iou_metric = tm.JaccardIndex(task="binary").to(device)
prec = tm.Precision(task="binary").to(device)
rec = tm.Recall(task="binary").to(device)

model.eval()

with torch.no_grad():
    for inputs, targets in test_dl:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        acc.update(outputs, targets)
        f1.update(outputs, targets)
        iou_metric.update(torch.argmax(outputs, dim=1), targets)
        prec.update(torch.argmax(outputs, dim=1), targets)
        rec.update(torch.argmax(outputs, dim=1), targets)

acc = acc.compute()
f1 = f1.compute()
m_iou = iou_metric.compute()
prec = prec.compute()
rec = rec.compute()

print(f'Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}, Test IoU: {m_iou:.4f}')
print(f'Test Precision: {prec:.4f}, Test Recall: {rec:.4f}')

#Постороение Penta-diagram
num_vars = 5

values = [acc, f1, m_iou, prec, rec]
values += values[:1]  # Замкнуть график

# Углы для вершин пятиугольника
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Замыкание графика

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.fill(angles, values, color='b', alpha=0.3)
ax.plot(angles, values, color='b', linewidth=2)

labels = ['Accuracy', 'F1', 'IoU', 'Precision', 'Recall']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_yticklabels([])
ax.set_title("Penta-Diagram", fontsize=14)

plt.show()

