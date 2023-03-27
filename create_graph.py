import matplotlib.pyplot as plt

# Данные
t = [1, 2, 3, 4]

q = [10, 20, 50, 100, 500, 1000, 2000]

data = [
    [0, 0, 1, 11, 1426, 11877, 118857],
    [0, 0, 0, 5, 734, 6590, 58937],
    [0, 0, 0, 4, 505, 4492, 39280],
    [0, 0, 0, 3, 368, 3464, 29652]
]



# Создание графика
fig, ax = plt.subplots()
for i, row in enumerate(data):
    ax.plot(q, row, marker='o', label=f'{t[i]} поток(а/ов)')

# Настройка графика
ax.set_title('Время выполнения программы \nв зависимости от размера матрицы и количества потоков')
ax.set_xlabel('Размер матрицы (q)')
ax.set_ylabel('Время выполнения (мс)')
ax.legend()
ax.grid()

# Отображение графика
plt.savefig('row.png', dpi=400)

plt.show()