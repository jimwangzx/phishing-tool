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
      	content: 'Please write an essay about your favorite DOM element.',
    	subject_line: "",
		email_address: "",
		phishing: JSON.parse(localStorage.getItem('phishing')) || 0,
		notPhishing:JSON.parse(localStorage.getItem('notPhishing')) || 100
		};
 		const accessToken = "access";
    	this.apiClient = new APIClient(accessToken);

    	this.handleChange = this.handleChange.bind(this);
		this.handleSubjChange = this.handleSubjChange.bind(this);
		this.handleEmailChange = this.handleEmailChange.bind(this);
    	this.handleSubmit = this.handleSubmit.bind(this);
  	}
 render() {  
	return (
    <div style={{ marginLeft: "40%"}}>
      	<h2>Phishing detector</h2>
		
		<div className="multiline" style={{marginRight: "40%"}}>
        <form onSubmit={this.handleSubmit}>
		<label className='label'>Content</label>
		<TextBoxComponent multiline={true} onChange={this.handleChange} input={this.onInput = this.onInput.bind(this)} created={this.onCreate = this.onCreate.bind(this)} placeholder='Enter the text' floatLabelType='Never' ref = {scope => {this.textareaObj = scope }}/>
		     <label className='label'>Subject-line (optional)</label>
            <TextBoxComponent multiline={false} onChange={this.handleSubjChange}  placeholder='Enter the subject-line' floatLabelType='Never'/>
            <label className='label'>Email Address (optional)</label>
            <TextBoxComponent multiline={false} onChange={this.handleEmailChange} placeholder='Enter the email address' floatLabelType='Never'/> 
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
			['Phishing', this.state.phishing],
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
      		"content": this.state.content,
			"subject_line" : this.state.subject_line,
			"email_address" : this.state.email_address
    	}
	this.apiClient.doText(value).then((data) =>
    {   this.setState({phishing: data});
		this.setState({notPhishing: 100 - data});
		localStorage.setItem('phishing', data);
    	localStorage.setItem('notPhishing', 100 - data);

		 });
	}
	handleChange(event) {
    	this.setState({content: event.target.value});
  	}
	handleSubjChange(event) {
		this.setState({subject_line: event.target.value});
	}
	handleEmailChange(event) {
		this.setState({email_address: event.target.value});
	}
}
ReactDOM.render(<App />, document.getElementById('root'));

