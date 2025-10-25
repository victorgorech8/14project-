import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)

def create_individual_charts(stock_file):
    """Создает диаграммы для отдельной акции"""
    try:
        df = pd.read_csv(stock_file)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df = df.sort_values('TRADEDATE')
        
        stock_name = os.path.basename(stock_file).replace('.csv', '')
        
        # Создаем фигуру с несколькими subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Анализ акции {stock_name}', fontsize=16, fontweight='bold')
        
        # 1. График цен (OHLC)
        axes[0, 0].plot(df['TRADEDATE'], df['CLOSE'], label='Close', linewidth=2, color='blue')
        axes[0, 0].plot(df['TRADEDATE'], df['HIGH'], label='High', alpha=0.7, color='green')
        axes[0, 0].plot(df['TRADEDATE'], df['LOW'], label='Low', alpha=0.7, color='red')
        axes[0, 0].set_title('Динамика цен')
        axes[0, 0].set_ylabel('Цена')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Объемы торгов
        axes[0, 1].bar(df['TRADEDATE'], df['VOLUME'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Объемы торгов')
        axes[0, 1].set_ylabel('Объем')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Количество сделок
        axes[1, 0].plot(df['TRADEDATE'], df['NUMTRADES'], color='purple', linewidth=2)
        axes[1, 0].set_title('Количество сделок')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Волатильность (High - Low)
        df['VOLATILITY'] = df['HIGH'] - df['LOW']
        axes[1, 1].bar(df['TRADEDATE'], df['VOLATILITY'], alpha=0.7, color='red')
        axes[1, 1].set_title('Волатильность (High - Low)')
        axes[1, 1].set_ylabel('Волатильность')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'charts/individual/{stock_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Созданы диаграммы для: {stock_name}")
        
    except Exception as e:
        print(f"Ошибка при обработке {stock_file}: {e}")

def create_summary_charts(all_data):
    """Создает сводные диаграммы по всем акциям"""
    
    # Сводный график доходностей
    plt.figure(figsize=(15, 8))
    
    for stock in all_data['SECID'].unique()[:20]:  # Ограничиваем для читаемости
        stock_data = all_data[all_data['SECID'] == stock].sort_values('TRADEDATE')
        if len(stock_data) > 1:
            first_price = stock_data.iloc[0]['CLOSE']
            stock_data['RETURN'] = (stock_data['CLOSE'] - first_price) / first_price * 100
            plt.plot(stock_data['TRADEDATE'], stock_data['RETURN'], label=stock, alpha=0.7, linewidth=1)
    
    plt.title('Сравнение доходности акций (нормализовано)', fontsize=14, fontweight='bold')
    plt.ylabel('Доходность (%)')
    plt.xlabel('Дата')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/summary/returns_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Топ-10 акций по объему торгов
    volume_by_stock = all_data.groupby('SECID')['VOLUME'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    volume_by_stock.head(10).plot(kind='bar', color='skyblue')
    plt.title('Топ-10 акций по объему торгов', fontsize=14, fontweight='bold')
    plt.ylabel('Общий объем')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/summary/top10_volume.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Топ-10 акций по количеству сделок
    trades_by_stock = all_data.groupby('SECID')['NUMTRADES'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    trades_by_stock.head(10).plot(kind='bar', color='lightcoral')
    plt.title('Топ-10 акций по количеству сделок', fontsize=14, fontweight='bold')
    plt.ylabel('Количество сделок')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/summary/top10_trades.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(all_data):
    """Создает тепловую карту корреляций"""
    # Выбираем основные акции для корреляции
    top_stocks = all_data['SECID'].value_counts().head(15).index
    
    # Создаем pivot table с ценами закрытия
    pivot_data = all_data.pivot_table(
        index='TRADEDATE', 
        columns='SECID', 
        values='CLOSE'
    )
    
    # Берем только топ акции
    pivot_data = pivot_data[top_stocks]
    
    # Расчет корреляции
    correlation_matrix = pivot_data.corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Корреляция')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Тепловая карта корреляций акций', fontsize=14, fontweight='bold')
    
    # Добавляем значения корреляции
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('charts/summary/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Создаем папки для графиков
    os.makedirs('charts/individual', exist_ok=True)
    os.makedirs('charts/summary', exist_ok=True)
    
    # Находим все CSV файлы
    csv_files = glob.glob('moex_data/*.csv')
    
    # Убираем stocks_list.csv из обработки
    csv_files = [f for f in csv_files if 'stocks_list' not in f]
    
    print(f"Найдено файлов с данными: {len(csv_files)}")
    
    # Загружаем все данные для сводных графиков
    all_data_list = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            all_data_list.append(df)
            
            # Создаем индивидуальные графики для каждой акции
            create_individual_charts(file)
            
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
    
    # Объединяем все данные
    if all_data_list:
        all_data = pd.concat(all_data_list, ignore_index=True)
        
        # Создаем сводные графики
        create_summary_charts(all_data)
        create_correlation_heatmap(all_data)
        
        print(f"\nОбработано акций: {all_data['SECID'].nunique()}")
        print(f"Всего записей: {len(all_data)}")
        print(f"Период данных: {all_data['TRADEDATE'].min()} - {all_data['TRADEDATE'].max()}")
    else:
        print("Нет данных для обработки")
    
    print("\nВсе диаграммы сохранены в папке 'charts/'")

if __name__ == "__main__":
    main()