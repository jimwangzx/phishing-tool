import logo from './logo.svg';
import './App.css';
import TextField from "@material-ui/core/TextField";
import { Chart } from "react-google-charts";
import { TextBoxComponent } from '@syncfusion/ej2-react-inputs';
import * as React from 'react';
import * as ReactDOM from "react-dom";

export default class App extends React.Component<{}, {}> {
    textareaObj: any;
	constructor(props) {
    	super(props);
    	this.state = {
      	value: 'Please write an essay about your favorite DOM element.'
    	};

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
		 <input type="submit" value="Submit" />
		</form>
		</div>
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
    	alert('An essay was submitted: ' + this.state.value);
    	event.preventDefault();
  	}
	handleChange(event) {
    	this.setState({value: event.target.value});
  	}
}
ReactDOM.render(<App />, document.getElementById('root'));

