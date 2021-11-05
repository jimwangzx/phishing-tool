import { combineReducers } from 'redux'
import phishingReducer from './phishingReducer'

const allReducers =  combineReducers({
   phishing: phishingReducer
})

export default allReducers;
