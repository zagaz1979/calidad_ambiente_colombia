import matplotlib.pyplot as plt
import seaborn as sns

def plot_line_evolucion(df, x_col, y_col, hue_col, title, ylabel, xlabel="Año", streamlit_mode=False):
    """
    Gráfico de evolución temporal (línea) por categoría.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        marker='o'
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode:
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()
    else:
        plt.show()

def plot_box_distribution(df, x_col, y_col, hue_col, title, ylabel, xlabel, streamlit_mode=False):
    """
    Gráfico de cajas para ver la distribución de valores por categoría.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode:
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()
    else:
        plt.show()

def plot_bar_categorico(df, x_col, y_col, hue_col, title, ylabel, xlabel, palette="Set2", streamlit_mode=False):
    """
    Gráfico de barras para comparar promedios o frecuencias por categoría.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode:
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()
    else:
        plt.show()

def plot_count_categorico(df, x_col, hue_col, title, ylabel="Cantidad", xlabel="", order=None, palette="Set2", streamlit_mode=False):
    """
    Gráfico de conteo para ver la frecuencia de categorías.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x=x_col,
        hue=hue_col,
        order=order,
        palette=palette
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode:
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()
    else:
        plt.show()

def plot_bar_horizontal(df, x_col, y_col, title, xlabel, ylabel, palette="Reds_r", streamlit_mode=False):
    """
    Gráfico de barras horizontal (ej. Top 10 municipios con peor calidad).
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        palette=palette
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if streamlit_mode:
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()
    else:
        plt.show()