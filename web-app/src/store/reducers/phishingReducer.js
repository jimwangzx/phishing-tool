import {POST_PHISHING} from '../types';


const phishingReducer = (state = 0, action) => {

	switch(action.type){
		case POST_PHISHING:	
			return action.payload
		default: return state
	}
}
export default phishingReducer;
