import pandas as pd
import matplotlib.pyplot as plt





def main() -> None:
    df = pd.read_csv(r"dl_for_nlp/Data/Churn_Modelling.csv")
    print(df.head())



    # Bar plot: Credit Score by Surname
    plt.bar(df['Surname'], df['CreditScore'])
    plt.xlabel('Surname')
    plt.ylabel('Credit Score')
    plt.title('Credit Score by Customer')
    plt.xticks(rotation=45)
    plt.show()

    # Pie chart: Gender distribution
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.show()

    # Scatter plot: Age vs Estimated Salary
    plt.scatter(df['Age'], df['EstimatedSalary'])
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.title('Age vs Estimated Salary')
    plt.show()


if __name__ == "__main__":
    main()
