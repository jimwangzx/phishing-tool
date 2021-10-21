import logo from './logo.svg';
import './App.css';
import React, { useState } from "react";
import TextField from "@material-ui/core/TextField";
import { Chart } from "react-google-charts";

function App() {
  const [name, setName] = useState("");
  
return (
    <div
      style={{
        marginLeft: "40%",
      }}
    >
      <h2>Anti-Phishing</h2>
      <TextField
        value={name}
        label="Enter a text"
        onChange={(e) => {
          setName(e.target.value);
        }}
      />
      <h3>Your Enter Value is: {name} </h3>
<Chart
  width={'500px'}
  height={'300px'}
  chartType="PieChart"
  loader={<div>Loading Chart</div>}
  data={[
    ['Task', 'Hours per Day'],
	['Phinsing', 2],
    ['Not Phishing', 7],
  ]}
  options={{
    title: 'Is it phishing?',
  }}
  rootProps={{ 'data-testid': '1' }}
/>
    </div>
  );
};

export default App;
