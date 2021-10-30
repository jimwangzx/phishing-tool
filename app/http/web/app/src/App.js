import logo from './logo.svg';
import './App.css';
import TextField from "@material-ui/core/TextField";
import { Chart } from "react-google-charts";
import { TextBoxComponent } from '@syncfusion/ej2-react-inputs';
import axios from "axios"; 
import * as React from 'react';
import * as ReactDOM from "react-dom";
import APIClient from './apiClient'

export default class App extends React.Component<{}, {}> {
    textareaObj: any;
	constructor(props) {
    	super(props);
    	this.state = {
      	value: 'Please write an essay about your favorite DOM element.',
    	phishing: JSON.parse(localStorage.getItem('phishing')) || 0,
		notPhishing:JSON.parse(localStorage.getItem('notPhishing')) || 100
		};
 		const accessToken = "access";
    	this.apiClient = new APIClient(accessToken);

    	this.handleChange = this.handleChange.bind(this);
    	this.handleSubmit = this.handleSubmit.bind(this);
  	}
 render() {  
	return (
    <div style={{ marginLeft: "40%"}}>
      	<h2>Anti-Phishing</h2>
		
		<div className="multiline" style={{marginRight: "40%"}}>
        <form onSubmit={this.handleSubmit}>
		<TextBoxComponent multiline={true} onChange={this.handleChange} input={this.onInput = this.onInput.bind(this)} created={this.onCreate = this.onCreate.bind(this)} placeholder='Enter your text' floatLabelType='Auto' ref = {scope => {this.textareaObj = scope }}/>
		 <input type="submit" value="Submit" style={{marginLeft: "40%"}} />
		</form>
		</div>
		<Chart
  			width={'500px'}
  			height={'300px'}
  			chartType="PieChart"
  			loader={<div>Loading Chart</div>}
  			data={[
   			 ['Task', 'Hours per Day'],
			['Phinsing', this.state.phishing],
    		['Not Phishing', this.state.notPhishing],
  			]}
  			options={{
    		title: 'Is it phishing?',
  			}}
  			rootProps={{ 'data-testid': '1' }}
		/>
    </div>
  );
}
     onCreate(): void {
        this.textareaObj.addAttributes({rows: 1});
        this.textareaObj.respectiveElement.style.height = "auto";
        this.textareaObj.respectiveElement.style.height = (this.textareaObj.respectiveElement .scrollHeight)+"px";
    }
    onInput(): void {
        this.textareaObj.respectiveElement.style.height = "auto";
        this.textareaObj.respectiveElement.style.height = (this.textareaObj.respectiveElement .scrollHeight)+"px";
    }
  	handleSubmit(event) {
		const value = {
			"id": "1",
      		"text": this.state.value
    	}
	this.apiClient.doText(value).then((data) =>
    {   this.setState({phishing: data});
		this.setState({notPhishing: 100 - data});
		localStorage.setItem('phishing', data);
    	localStorage.setItem('notPhishing', 100 - data);

		 });
	}
	handleChange(event) {
    	this.setState({value: event.target.value});
  	}
}
ReactDOM.render(<App />, document.getElementById('root'));

