import streamlit as st
options = st.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])
print(options)
print(type(options))
print(len(options))
print(options[0])
st.write('You selected:', options)