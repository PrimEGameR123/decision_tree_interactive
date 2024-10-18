import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
from sklearn import tree

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array


options = ['Classifier', 'Regression']
st.sidebar.markdown("# Decision Tree")
radio_python = st.sidebar.radio('',options=options, index=0)

if radio_python == "Classifier":
    X, y = make_moons(n_samples=500, noise = 0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    plt.style.use('fivethirtyeight')
    st.sidebar.markdown("## Decision Tree Classifier")

    criterion = st.sidebar.selectbox(
        'Criterion',
        ('gini', 'entropy')
    )

    splitter = st.sidebar.selectbox(
        'Splitter',
        ('best', 'random')
    )

    max_depth = int(st.sidebar.number_input('Max Depth'))

    min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

    max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1236)

    max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

    # Load initial graph
    fig, ax = plt.subplots()

    # Plot initial graph
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    orig = st.pyplot(fig)

    if st.sidebar.button('Run Algorithm'):

        orig.empty()

        if max_depth == 0:
            max_depth = None

        if max_leaf_nodes == 0:
            max_leaf_nodes = None

        clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        XX, YY, input_array = draw_meshgrid()
        labels = clf.predict(input_array)

        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig = st.pyplot(fig)
        st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))

        tree = export_graphviz(clf,feature_names=["Col1","Col2"])

        st.graphviz_chart(tree)

if radio_python == "Regression":
    rng = np.random.RandomState(0)

    n_sample = 1000
    data_max, data_min = 1.4, -1.4
    len_data = data_max - data_min
    # sort the data to make plotting easier later
    data = np.sort(rng.rand(n_sample) * len_data - len_data / 2)
    noise = rng.randn(n_sample) * 0.3
    y = data**3 - 0.5 * data**2 + noise
    X = data.reshape([1000,1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    plt.style.use('fivethirtyeight')

    st.sidebar.markdown("# Decision Tree Regression")

    criterion = st.sidebar.selectbox(
        'Criterion',
        ('squared_error', 'absolute_error')
    )

    splitter = st.sidebar.selectbox(
        'Splitter',
        ('best', 'random')
    )

    max_depth = int(st.sidebar.number_input('Max Depth'))

    min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)


    max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

    # Load initial graph
    fig, ax = plt.subplots()

    # Plot initial graph
    ax.scatter(X, y, c=y, cmap='viridis')
    orig = st.pyplot(fig)

    if st.sidebar.button('Run Algorithm'):

        orig.empty()

        if max_depth == 0:
            max_depth = None

        if max_leaf_nodes == 0:
            max_leaf_nodes = None

        clf = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # XX, YY, input_array = draw_meshgrid()
        # labels = clf.predict(input_array)

        ax.scatter(X, y, c=y, cmap='viridis')
        X_grid = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
        y_grid = clf.predict(X_grid)
        ax.plot(X_grid, y_grid, color='black', label='prediction', linewidth=1)
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig = st.pyplot(fig)
        st.subheader("R-squared for Decision Tree  " + str(round(r2_score(y_test, y_pred), 2)))

        tree = export_graphviz(clf, filled=True, rounded=True)

        st.graphviz_chart(tree)
